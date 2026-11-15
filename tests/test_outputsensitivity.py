import pytest
import torch
from outputsensitivity import OutputSensitivity, create_outputsensitivity

class TestOutputSensitivity:
    def test_init(self):
        m = OutputSensitivity(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = OutputSensitivity(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_outputsensitivity()
        assert isinstance(m, OutputSensitivity)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
