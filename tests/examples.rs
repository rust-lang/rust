use std::fs::File;
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

            let _ = Command::new(exe_path.clone())
                .arg("--crate-type=lib")
                .arg(file)
                .env("RUST_SEMVERVER_TEST", "1")
                .env("RUST_BACKTRACE", "full")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(err)
                .status();

            println!("{:?}", exe_path);

            success &= Command::new("git")
                .args(&["diff", "--exit-code", out_file.to_str().unwrap()])
                .status()
                .unwrap()
                .success();
        }
    }

    assert!(success, "an error occured");
}
