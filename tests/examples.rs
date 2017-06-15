use std::fs::File;
use std::io::Write;
use std::os::unix::io::{AsRawFd, FromRawFd};
use std::process::{Command, Stdio};

#[test]
fn examples() {
    let mut success = true;

    for file in std::fs::read_dir("tests/examples")
            .expect("could not read dir")
            .map(|f| f.expect("could not get file info").path()) {
        if file.extension().map_or(false, |e| e == "rs") {
            let out_file = file.with_extension("out");
            let output = File::create(&out_file).expect("could not create file");
            let fd = output.as_raw_fd();
            //let out = unsafe {Stdio::from_raw_fd(fd)};
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
                    .stdout(out)
                    .stderr(err)
                    .spawn()
                    .expect("could not run rust-semverver");

                if let Some(ref mut stdin) = child.stdin {
                    stdin
                        .write_fmt(format_args!("extern crate oldandnew;"))
                        .expect("could not pipe to rust-semverver");
                } else {
                    panic!("could not pipe to rustc (wtf?)");
                }

                success &= child.wait().expect("could not wait for child").success();

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

    assert!(success, "an error occured");
}
