use std::env;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

#[test]
fn debug() {
    if let Ok(path) = env::var("LD_LIBRARY_PATH") {
        let mut dump =
            File::create(Path::new("tests/debug.sh")).expect("could not create dump file");

        let metadata = dump.metadata().expect("could not access dump file metadata");
        let mut permissions = metadata.permissions();
        permissions.set_mode(0o755);
        let _ = dump.set_permissions(permissions);

        let _ = writeln!(dump, r#"#!/bin/sh
export PATH=./target/debug:$PATH
export LD_LIBRARY_PATH={}
export RUST_BACKTRACE=full
export RUST_SEMVER_CRATE_VERSION=1.0.0

if [ "$1" = "-s" ]; then
    shift
    arg_str="set args --crate-type=lib $(cargo semver "$@") tests/helper/test.rs"
else
    rustc --crate-type=lib -o libold.rlib "$1/old.rs"
    rustc --crate-type=lib -o libnew.rlib "$1/new.rs"
    arg_str="set args --crate-type=lib --extern old=libold.rlib --extern new=libnew.rlib tests/helper/test.rs"
fi

export RUST_LOG=debug

src_str="set substitute-path /checkout $(rustc --print sysroot)/lib/rustlib/src/rust"

rust-gdb ./target/debug/rust-semverver -iex "$arg_str" -iex "$src_str"

rm lib*.rlib
"#, path);
    }
}
