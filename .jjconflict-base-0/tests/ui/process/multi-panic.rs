//@ run-pass
//@ needs-subprocess
//@ needs-unwind
//@ ignore-backends: gcc

fn check_for_no_backtrace(test: std::process::Output) {
    assert!(!test.status.success());
    let err = String::from_utf8_lossy(&test.stderr);
    let mut it = err.lines().filter(|l| !l.is_empty());

    assert_eq!(
        it.next().map(|l| l.starts_with("thread '<unnamed>' (") && l.contains("panicked")),
        Some(true),
        "out: ```{err}```",
    );
    assert_eq!(it.next().is_some(), true, "out: ```{err}```");
    assert_eq!(
        it.next(),
        Some(
            "note: run with `RUST_BACKTRACE=1` \
                                environment variable to display a backtrace"
        ),
        "out: ```{err}```",
    );
    assert_eq!(
        it.next().map(|l| l.starts_with("thread 'main' (") && l.contains("panicked at")),
        Some(true),
        "out: ```{err}```",
    );
    assert_eq!(it.next().is_some(), true, "out: ```{err}```");
    assert_eq!(it.next(), None, "out: ```{err}```");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "run_test" {
        let _ = std::thread::spawn(|| {
            panic!();
        })
        .join();

        panic!();
    } else {
        let test = std::process::Command::new(&args[0])
            .arg("run_test")
            .env_remove("RUST_BACKTRACE")
            .output()
            .unwrap();
        check_for_no_backtrace(test);
        let test = std::process::Command::new(&args[0])
            .arg("run_test")
            .env("RUST_BACKTRACE", "0")
            .output()
            .unwrap();
        check_for_no_backtrace(test);
    }
}
