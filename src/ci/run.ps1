
if (Test-Path env:\CI_JOB_NAME) {
    Write-Host "[CI_JOB_NAME=$env:CI_JOB_NAME]"
}

. $PSScriptRoot/shared.ps1

if (($env:TRAVIS -ne 'true') -or ($env:TRAVIS_BRANCH -eq 'auto')) {
    $env:RUST_CONFIGURE_ARGS = "$env:RUST_CONFIGURE_ARGS --set build.print-step-timings --enable-verbose-tests"
}

$env:RUST_CONFIGURE_ARGS = "$env:RUST_CONFIGURE_ARGS --enable-sccache"
$env:RUST_CONFIGURE_ARGS = "$env:RUST_CONFIGURE_ARGS --disable-manage-submodules"
$env:RUST_CONFIGURE_ARGS = "$env:RUST_CONFIGURE_ARGS --enable-locked-deps"
$env:RUST_CONFIGURE_ARGS = "$env:RUST_CONFIGURE_ARGS --enable-cargo-native-static"
$env:RUST_CONFIGURE_ARGS = "$env:RUST_CONFIGURE_ARGS --set rust.codegen-units-std=1"

if ([string]::IsNullOrEmpty($env:DIST_SRC)) {
    $env:RUST_CONFIGURE_ARGS = "$env:RUST_CONFIGURE_ARGS --disable-dist-src"
}

$env:RUST_RELEASE_CHANNEL = 'nightly'

if (-not [string]::IsNullOrEmpty("${env:DEPLOY}${env:DEPLOY_ALT}")) {
    $env:RUST_CONFIGURE_ARGS = "$env:RUST_CONFIGURE_ARGS --release-channel=$RUST_RELEASE_CHANNEL"
    $env:RUST_CONFIGURE_ARGS = "$env:RUST_CONFIGURE_ARGS --enable-llvm-static-stdcpp"
    $env:RUST_CONFIGURE_ARGS = "$env:RUST_CONFIGURE_ARGS --set rust.remap-debuginfo"

    if ($env:NO_LLVM_ASSERTIONS -eq "1") {
        $env:RUST_CONFIGURE_ARGS = "$env:RUST_CONFIGURE_ARGS --disable-llvm-assertions"

    }
    elseif ( -not [string]::IsNullOrEmpty($env:DEPLOY_ALT) ) {
        $env:RUST_CONFIGURE_ARGS = "$env:RUST_CONFIGURE_ARGS --enable-llvm-assertions"
        $env:RUST_CONFIGURE_ARGS = "$env:RUST_CONFIGURE_ARGS --set rust.verify-llvm-ir"
    }
}
else {
    if ([string]::IsNullOrEmpty($env:NO_DEBUG_ASSERTIONS)) {
        $env:RUST_CONFIGURE_ARGS = "$env:RUST_CONFIGURE_ARGS --enable-debug-assertions"
    }

    # In general we always want to run tests with LLVM assertions enabled, but not
    # all platforms currently support that, so we have an option to disable.
    if ([string]::IsNullOrEmpty($env:NO_LLVM_ASSERTIONS)) {
        RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --enable-llvm-assertions"
    }

    $env:RUST_CONFIGURE_ARGS = "$env:RUST_CONFIGURE_ARGS --set rust.verify-llvm-ir"
}
 
if  (($env:RUST_RELEASE_CHANNEL -eq "nightly") -or ([string]::IsNullOrEmpty($env:DIST_REQUIRE_ALL_TOOLS))) {
    $env:RUST_CONFIGURE_ARGS="$env:RUST_CONFIGURE_ARGS --enable-missing-tools"
}

$env:SCCACHE_IDLE_TIMEOUT=10800 
sccache --start-server

if (-not [string]::IsNullOrEmpty($env:RUN_CHECK_WITH_PARALLEL_QUERIES) {
  $env:SRC/configure --enable-parallel-compiler
  $env:CARGO_INCREMENTAL=0 
  python2.7 ../x.py check
  rm -force config.toml
  rm -recurse -force build
}

$env:SRC/configure $env:RUST_CONFIGURE_ARGS
retry make prepare
make check-bootstrap


$ncpus = (get-wmiobject win32_processor).NumberOfLogicalProcessors

if ((test-path env:\SCRIPT) -and  (-not [string]::IsNullOrEmpty($env:SCRIPT))){
    & $env:SCRIPT
} 
else {
    make -j $ncpus tidy
    make -j $ncpus all
    make -j $ncpus $env:RUST_CHECK_TARGET
}
