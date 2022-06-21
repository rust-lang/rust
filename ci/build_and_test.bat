set "RUSTFLAGS=-D warnings"
set "RUSTFMT_CI=1"

:: Print version information
rustc -Vv || exit /b 1
cargo -V || exit /b 1

:: Build and test main crate
cargo build --locked || exit /b 1
cargo test || exit /b 1

:: Build and test other crates
cd config_proc_macro || exit /b 1
cargo build --locked || exit /b 1
cargo test || exit /b 1
