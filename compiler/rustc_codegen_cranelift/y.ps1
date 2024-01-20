$ErrorActionPreference = "Stop"

$host.ui.WriteErrorLine("[BUILD] build system")
New-Item -ItemType Directory -Force -Path build | Out-Null
& rustc build_system/main.rs -o build\y.exe -Cdebuginfo=1 --edition 2021
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
& build\y.exe $args
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
