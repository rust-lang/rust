cargo build || exit 1

rustc -Zcodegen-backend=$(pwd)/target/debug/librustc_codegen_cranelift.so example.rs --crate-type lib -Og
rustc -Zcodegen-backend=$(pwd)/target/debug/librustc_codegen_cranelift.so ./target/libcore/src/libcore/lib.rs --crate-type lib -Og
