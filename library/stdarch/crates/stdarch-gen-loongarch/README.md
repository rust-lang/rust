# LoongArch LSX/LASX intrinsic code generator

A small tool that allows to quickly generate intrinsics for the LoongArch LSX/LASX architectures.

The specification for the intrinsics can be found in `lsx.spec` or `lasx.spec`.

To run and re-generate the code run the following from the root of the `stdarch` crate.

LSX:
```
# Generate bindings
OUT_DIR=`pwd`/crates/stdarch-gen-loongarch cargo run -p stdarch-gen-loongarch -- crates/stdarch-gen-loongarch/lsxintrin.h
OUT_DIR=`pwd`/crates/core_arch cargo run -p stdarch-gen-loongarch -- crates/stdarch-gen-loongarch/lsx.spec

# Generate tests
OUT_DIR=`pwd`/crates/stdarch-gen-loongarch cargo run -p stdarch-gen-loongarch -- crates/stdarch-gen-loongarch/lsx.spec test
loongarch64-unknown-linux-gnu-gcc -static -o lsx crates/stdarch-gen-loongarch/lsx.c -mlasx -mfrecipe
qemu-loongarch64 ./lsx > crates/core_arch/src/loongarch64/lsx/tests.rs
rustfmt crates/core_arch/src/loongarch64/lsx/tests.rs
```

LASX:
```
# Generate bindings
OUT_DIR=`pwd`/crates/stdarch-gen-loongarch cargo run -p stdarch-gen-loongarch -- crates/stdarch-gen-loongarch/lasxintrin.h
OUT_DIR=`pwd`/crates/core_arch cargo run -p stdarch-gen-loongarch -- crates/stdarch-gen-loongarch/lasx.spec

# Generate tests
OUT_DIR=`pwd`/crates/stdarch-gen-loongarch cargo run -p stdarch-gen-loongarch -- crates/stdarch-gen-loongarch/lasx.spec test
loongarch64-unknown-linux-gnu-gcc -static -o lasx crates/stdarch-gen-loongarch/lasx.c -mlasx -mfrecipe
qemu-loongarch64 ./lasx > crates/core_arch/src/loongarch64/lasx/tests.rs
rustfmt crates/core_arch/src/loongarch64/lasx/tests.rs
```
