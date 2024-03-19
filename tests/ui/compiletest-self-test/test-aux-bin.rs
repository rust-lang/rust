//@ ignore-cross-compile because we run the compiled code
//@ aux-bin: print-it-works.rs
//@ run-pass

fn main() {
    let stdout =
        std::process::Command::new("auxiliary/bin/print-it-works").output().unwrap().stdout;
    assert_eq!(stdout, b"it works\n");
}
