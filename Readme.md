# The Enzyme Automatic differentiator

To build Enzyme run the following steps:
```bash
cd enzyme && mkdir build && cd build
cmake ..
make
```

This will pick up your default installation of LLVM, to specify a specific LLVM installation add the LLVM_DIR flag to cmake as follows:

```bash
cmake .. -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm
```

To run Enzyme tests use the following:
```bash
make check-enzyme
```

To run C++ integration tests (which require boost/eigen and take quite a while) use the following:
```bash
make check-enzyme-integration
```
