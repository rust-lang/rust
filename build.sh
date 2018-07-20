cargo build || exit 1

rustc -Zcodegen-backend=$(pwd)/target/debug/librustc_codegen_cranelift.so mini_core.rs --crate-name mini_core --crate-type lib -Og &&
rustc -Zcodegen-backend=$(pwd)/target/debug/librustc_codegen_cranelift.so -L crate=. example.rs --crate-type lib -Og &&
rustc -Zcodegen-backend=$(pwd)/target/debug/librustc_codegen_cranelift.so ./target/libcore/src/libcore/lib.rs --crate-type lib -Og

rm libmini_core.rlib libexample.rlib
