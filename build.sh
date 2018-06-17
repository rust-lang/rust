cargo build || exit 1

rustc -Zcodegen-backend=$(pwd)/target/debug/librustc_codegen_cretonne.so example.rs --crate-type lib
