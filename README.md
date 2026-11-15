# OutputSensitivity

Rank neural-network layers by how much perturbing them moves the model output.

## What it does

Given a model dict {name: 2-D matrix} applied in insertion order, the
module provides two views:

- Spectral: singular spectra, effective rank, condition number, and
  find_bottlenecks() (smallest effective rank first).
- Output sensitivity (the namesake feature): output_sensitivity() adds
  Gaussian noise to one layer at a time and measures the relative change
  of the final output, most sensitive layer first.

    from outputsensitivity import output_sensitivity

    for row in output_sensitivity(model, x, noise_std=0.01):
        print(row["layer"], round(row["mean_relative_change"], 5))

## API

- layer_spectrum, effective_rank, layer_report, analyze_model,
  find_bottlenecks
- forward(layers, x) - applies layers in insertion order
- output_sensitivity(layers, x, noise_std, repeats, rng)

## Honest results and limitations

- Sensitivity is a finite-difference estimate under random noise, not a
  gradient; it varies between runs and depends on the input vector.
- Ranking by mean relative change says which layer moves the output, not
  which layer is "important" for accuracy - those are different questions.
- No nonlinearity: the forward pass is pure matrix products, so this
  analyses linear maps only.
- Negative result from the source experiments: spectral rank and output
  sensitivity do not always agree - a low-rank layer can still move the
  output a lot. Use both views, not just one.

## Status

Part of the kapildhinwa2010-dev toolkit experiments. CI and CodeQL run on
every push to main. License: MIT.
