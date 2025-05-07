set "RUSTFLAGS=-D warnings"
set "RUSTFMT_CI=1"

:: Print version information
rustc -Vv || exit /b 1
cargo -V || exit /b 1

:: Build and test main crate
if "%CFG_RELEASE_CHANNEL%"=="nightly" (
    cargo build --locked --all-features || exit /b 1
) else (
    cargo build --locked || exit /b 1
)
cargo test || exit /b 1

:: Build and test config_proc_macro
cd config_proc_macro || exit /b 1
cargo build --locked || exit /b 1
cargo test || exit /b 1

:: Build and test check_diff
cd ..
cd check_diff || exit /b 1
cargo build --locked || exit /b 1
cargo test || exit /b 1
