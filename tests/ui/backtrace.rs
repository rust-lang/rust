// run-pass
// ignore-android FIXME #17520
// ignore-emscripten spawning processes is not supported
// ignore-openbsd no support for libbacktrace without filename
// ignore-sgx no processes
// ignore-msvc see #62897 and `backtrace-debuginfo.rs` test
// ignore-fuchsia Backtraces not symbolized
// compile-flags:-g
// compile-flags:-Cstrip=none

use std::env;
use std::process::{Command, Stdio};
use std::str;

#[inline(never)]
fn foo() {
    let _v = vec![1, 2, 3];
    if env::var_os("IS_TEST").is_some() {
        panic!()
    }
}

#[inline(never)]
fn double() {
    struct Double;

    impl Drop for Double {
        fn drop(&mut self) { panic!("twice") }
    }

    let _d = Double;

    panic!("once");
}

fn template(me: &str) -> Command {
    let mut m = Command::new(me);
    m.env("IS_TEST", "1")
     .stdout(Stdio::piped())
     .stderr(Stdio::piped());
    return m;
}

fn expected(fn_name: &str) -> String {
    format!(" backtrace::{}", fn_name)
}

#[cfg(not(panic = "abort"))]
fn contains_verbose_expected(s: &str, fn_name: &str) -> bool {
    // HACK(eddyb) work around the fact that verbosely demangled stack traces
    // (from `RUST_BACKTRACE=full`, or, as is the case here, panic-in-panic)
    // may contain symbols with hashes in them, i.e. `backtrace[...]::`.
    let prefix = " backtrace";
    let suffix = &format!("::{}", fn_name);
    s.match_indices(prefix).any(|(i, _)| {
        s[i + prefix.len()..]
            .trim_start_matches('[')
            .trim_start_matches(char::is_alphanumeric)
            .trim_start_matches(']')
            .starts_with(suffix)
    })
}

fn runtest(me: &str) {
    // Make sure that the stack trace is printed
    let p = template(me).arg("fail").env("RUST_BACKTRACE", "1").spawn().unwrap();
    let out = p.wait_with_output().unwrap();
    assert!(!out.status.success());
    let s = str::from_utf8(&out.stderr).unwrap();
    assert!(s.contains("stack backtrace") && s.contains(&expected("foo")),
            "bad output: {}", s);
    assert!(s.contains(" 0:"), "the frame number should start at 0");

    // Make sure the stack trace is *not* printed
    // (Remove RUST_BACKTRACE from our own environment, in case developer
    // is running `make check` with it on.)
    let p = template(me).arg("fail").env_remove("RUST_BACKTRACE").spawn().unwrap();
    let out = p.wait_with_output().unwrap();
    assert!(!out.status.success());
    let s = str::from_utf8(&out.stderr).unwrap();
    assert!(!s.contains("stack backtrace") && !s.contains(&expected("foo")),
            "bad output2: {}", s);

    // Make sure the stack trace is *not* printed
    // (RUST_BACKTRACE=0 acts as if it were unset from our own environment,
    // in case developer is running `make check` with it set.)
    let p = template(me).arg("fail").env("RUST_BACKTRACE","0").spawn().unwrap();
    let out = p.wait_with_output().unwrap();
    assert!(!out.status.success());
    let s = str::from_utf8(&out.stderr).unwrap();
    assert!(!s.contains("stack backtrace") && !s.contains(" - foo"),
            "bad output3: {}", s);

    #[cfg(not(panic = "abort"))]
    {
        // Make sure a stack trace is printed
        let p = template(me).arg("double-fail").spawn().unwrap();
        let out = p.wait_with_output().unwrap();
        assert!(!out.status.success());
        let s = str::from_utf8(&out.stderr).unwrap();
        // loosened the following from double::h to double:: due to
        // spurious failures on mac, 32bit, optimized
        assert!(s.contains("stack backtrace") && contains_verbose_expected(s, "double"),
                "bad output3: {}", s);

        // Make sure a stack trace isn't printed too many times
        //
        // Currently it is printed 3 times ("once", "twice" and "panic in a
        // function that cannot unwind") but in the future the last one may be
        // removed.
        let p = template(me).arg("double-fail")
                                    .env("RUST_BACKTRACE", "1").spawn().unwrap();
        let out = p.wait_with_output().unwrap();
        assert!(!out.status.success());
        let s = str::from_utf8(&out.stderr).unwrap();
        let mut i = 0;
        for _ in 0..3 {
            i += s[i + 10..].find("stack backtrace").unwrap() + 10;
        }
        assert!(s[i + 10..].find("stack backtrace").is_none(),
                "bad output4: {}", s);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 && args[1] == "fail" {
        foo();
    } else if args.len() >= 2 && args[1] == "double-fail" {
        double();
    } else {
        runtest(&args[0]);
    }
}
