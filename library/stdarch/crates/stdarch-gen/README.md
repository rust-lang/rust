# Neon intrinsic code generator

A small tool that allows to quickly generate intrinsics for the NEON architecture.

The specification for the intrinsics can be found in `neon.spec`.

To run and re-generate the code run the following from the root of the `stdarch` crate.

```
OUT_DIR=`pwd`/crates/core_arch cargo run -p stdarch-gen -- crates/stdarch-gen/neon.spec
```