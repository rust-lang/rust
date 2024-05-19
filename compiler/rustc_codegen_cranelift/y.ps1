$ErrorActionPreference = "Stop"

$host.ui.WriteErrorLine("[BUILD] build system")
& cargo run --manifest-path build_system/Cargo.toml -- $args
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
