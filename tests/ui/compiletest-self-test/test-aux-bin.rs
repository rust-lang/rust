//@ ignore-cross-compile because aux-bin does not yet support it
//@ ignore-remote because aux-bin does not yet support it
//@ aux-bin: print-it-works.rs
//@ run-pass

fn main() {
    let stdout =
        std::process::Command::new("auxiliary/bin/print-it-works").output().unwrap().stdout;
    assert_eq!(stdout, b"it works\n");
}
