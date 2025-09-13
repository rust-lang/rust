//! Test that with `-C panic=abort` the backtrace is not cut off by default
//! (i.e. without using `-C force-unwind-tables=yes`) by ensuring that our own
//! functions are in the backtrace. If we just check one function it might be
//! the last function, so make sure the backtrace can continue by checking for
//! two functions. Regression test for
//! <https://github.com/rust-lang/rust/issues/81902>.

//@ run-pass
//@ needs-subprocess
//@ compile-flags: -C panic=abort -C opt-level=0
//@ no-prefer-dynamic

static FN_1: &str = "this_function_must_be_in_the_backtrace";
fn this_function_must_be_in_the_backtrace() {
    and_this_function_too();
}

static FN_2: &str = "and_this_function_too";
fn and_this_function_too() {
    panic!("generate panic backtrace");
}

fn run_test() {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("whatever")
        .env("RUST_BACKTRACE", "full")
        .output()
        .unwrap();
    let backtrace = std::str::from_utf8(&output.stderr).unwrap();

    fn assert(function_name: &str, backtrace: &str) {
        assert!(
            backtrace.contains(function_name),
            "ERROR: no `{}` in stderr! actual stderr: {}",
            function_name,
            backtrace
        );
    }
    assert(FN_1, backtrace);
    assert(FN_2, backtrace);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 1 {
        run_test();
    } else {
        this_function_must_be_in_the_backtrace();
    }
}
