cargo build || exit 1

rustc -Zcodegen-backend=$(pwd)/target/debug/librustc_codegen_cranelift.dylib example.rs --crate-type lib -Og
rustc -Zcodegen-backend=$(pwd)/target/debug/librustc_codegen_cranelift.dylib ../rust_fork/src/libcore/lib.rs --crate-type lib -Og
