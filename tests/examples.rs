use std::env;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::io::{AsRawFd, FromRawFd};
use std::path::Path;
use std::process::{Command, Stdio};

#[test]
fn examples() {
    let mut success = true;

    let current_dir = env::current_dir().expect("could not determine current dir");
    let subst = format!("s#{}#$REPO_PATH#g", current_dir.to_str().unwrap());

    for file in std::fs::read_dir("tests/examples")
            .expect("could not read dir")
            .map(|f| f.expect("could not get file info").path()) {
        if file.extension().map_or(false, |e| e == "rs") {
            let out_file = file.with_extension("out");
            let output = File::create(&out_file).expect("could not create file");
            let fd = output.as_raw_fd();
            let err = unsafe { Stdio::from_raw_fd(fd) };
            let out = unsafe { Stdio::from_raw_fd(fd) };

            let compile_success = Command::new("rustc")
                .args(&["--crate-type=lib", "-o", "liboldandnew.rlib"])
                .arg(file)
                .env("RUST_BACKTRACE", "full")
                .stdin(Stdio::null())
                .status()
                .expect("could not run rustc")
                .success();

            success &= compile_success;

            if compile_success {
                let mut sed_child = Command::new("sed")
                    .arg(&subst)
                    .stdin(Stdio::piped())
                    .stdout(out)
                    .stderr(err)
                    .spawn()
                    .expect("could not run sed");

                let (err_pipe, out_pipe) = if let Some(ref mut stdin) = sed_child.stdin {
                    let fd = stdin.as_raw_fd();
                    unsafe { (Stdio::from_raw_fd(fd), Stdio::from_raw_fd(fd)) }
                } else {
                    panic!("could not pipe to sed");
                };

                let mut child = Command::new("./target/debug/rust-semverver")
                    .args(&["--crate-type=lib",
                            "--extern",
                            "oldandnew=liboldandnew.rlib",
                            "-"])
                    .env("RUST_SEMVERVER_TEST", "1")
                    .env("RUST_LOG", "debug")
                    .env("RUST_BACKTRACE", "full")
                    .env("RUST_SEMVER_CRATE_VERSION", "1.0.0")
                    .stdin(Stdio::piped())
                    .stdout(out_pipe)
                    .stderr(err_pipe)
                    .spawn()
                    .expect("could not run rust-semverver");

                if let Some(ref mut stdin) = child.stdin {
                    stdin
                        .write_fmt(format_args!("extern crate oldandnew;"))
                        .expect("could not pipe to rust-semverver");
                } else {
                    panic!("could not pipe to rustc");
                }

                success &= child.wait().expect("could not wait for child").success();
                success &= sed_child.wait().expect("could not wait for sed child").success();

                success &= Command::new("git")
                    .args(&["diff", "--exit-code", out_file.to_str().unwrap()])
                    .status()
                    .expect("could not run git diff")
                    .success();
            }
        }
    }

    Command::new("rm")
        .arg("liboldandnew.rlib")
        .status()
        .expect("could not run rm");

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
    arg_str="set args $(cargo semver "$@")"
    standalone=1
else
    rustc --crate-type=lib -o liboldandnew.rlib "$1"
    export RUST_SEMVERVER_TEST=1
    arg_str="set args --crate-type=lib --extern oldandnew=liboldandnew.rlib tests/helper/test.rs"
    standalone=0
fi

export RUST_LOG=debug

src_str="set substitute-path /checkout $(rustc --print sysroot)/lib/rustlib/src/rust"

rust-gdb ./target/debug/rust-semverver -iex "$arg_str" -iex "$src_str"

if [ $standalone = 0 ]; then
    rm liboldandnew.rlib
fi"#, path);
    }

    assert!(success, "an error occured");
}
