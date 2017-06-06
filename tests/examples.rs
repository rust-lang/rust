use std::fs::File;
use std::io::Write;
use std::os::unix::io::{AsRawFd, FromRawFd};
use std::path::Path;
use std::process::{Command, Stdio};

#[test]
fn examples() {
    let mut cur_path = std::env::current_dir().unwrap();
    loop {
        if let Some(fname) = cur_path.file_name() {
            if fname == "rust-semverver" {
                break;
            }
        }

        if !cur_path.pop() {
            panic!("blergh");
        }
    }

    let exe_path = cur_path.join(Path::new("target/debug/rust-semverver"));

    let mut success = true;

    // FIXME: this assumes a different dir
    for file in std::fs::read_dir("tests/examples").unwrap().map(|f| f.unwrap().path()) {
        if file.extension().map_or(false, |e| e == "rs") {
            let out_file = file.with_extension("out");
            let output = File::create(&out_file).unwrap();
            let fd = output.as_raw_fd();
            //let out = unsafe {Stdio::from_raw_fd(fd)};
            let err = unsafe {Stdio::from_raw_fd(fd)};
            let out = unsafe {Stdio::from_raw_fd(fd)}; 

            success &= Command::new("rustc")
                .args(&["--crate-type=lib", "-o", "liboldandnew.rlib"])
                .arg(file)
                .env("RUST_LOG", "debug")
                .env("RUST_BACKTRACE", "full")
                .stdin(Stdio::null())
                .status()
                .unwrap()
                .success();

            let mut child = Command::new(exe_path.clone())
                .args(&["--crate-type=lib",
                      "--extern",
                      "oldandnew=liboldandnew.rlib",
                      "-"])
                .env("RUST_SEMVERVER_TEST", "1")
                .env("RUST_LOG", "debug")
                .env("RUST_BACKTRACE", "full")
                .stdin(Stdio::piped())
                .stdout(out)
                .stderr(err)
                .spawn()
                .unwrap();

            if let Some(ref mut stdin) = child.stdin {
                stdin.write_fmt(format_args!("extern crate oldandnew;")).unwrap();
            } else {
                panic!("could not pipe to rustc (wtf?)");
            }

            success &= child
                .wait()
                .unwrap()
                .success();

            success &= Command::new("git")
                .args(&["diff", "--exit-code", out_file.to_str().unwrap()])
                .status()
                .unwrap()
                .success();
        }
    }

    Command::new("rm")
        .arg("liboldandnew.rlib")
        .status()
        .unwrap();

    assert!(success, "an error occured");
}
