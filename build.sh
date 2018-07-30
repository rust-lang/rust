cargo build || exit 1

cd examples/

RUSTC="rustc -Zcodegen-backend=$(pwd)/../target/debug/librustc_codegen_cranelift.so -Og -L crate=. --crate-type lib"

$RUSTC mini_core.rs --crate-name mini_core &&
$RUSTC example.rs &&
$RUSTC mini_core_hello_world.rs &&
$RUSTC ../target/libcore/src/libcore/lib.rs

rm *.rlib
