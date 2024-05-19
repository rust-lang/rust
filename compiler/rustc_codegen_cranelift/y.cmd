@echo off
echo [BUILD] build system >&2
cargo run --manifest-path build_system/Cargo.toml -- %* || goto :error
goto :EOF

:error
exit /b
