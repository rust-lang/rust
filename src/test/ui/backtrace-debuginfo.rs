// run-pass
// We disable tail merging here because it can't preserve debuginfo and thus
// potentially breaks the backtraces. Also, subtle changes can decide whether
// tail merging succeeds, so the test might work today but fail tomorrow due to a
// seemingly completely unrelated change.
// Unfortunately, LLVM has no "disable" option for this, so we have to set
// "enable" to 0 instead.

// compile-flags:-g -Cllvm-args=-enable-tail-merge=0 -Cllvm-args=-opt-bisect-limit=0
// compile-flags:-Cforce-frame-pointers=yes
// ignore-pretty issue #37195
// ignore-emscripten spawning processes is not supported
// ignore-sgx no processes
// normalize-stderr-test ".*\n" -> ""

// Note that above `-opt-bisect-limit=0` is used to basically disable
// optimizations. It creates tons of output on stderr, hence we normalize
// that away entirely.

use std::env;

#[path = "backtrace-debuginfo-aux.rs"] mod aux;

macro_rules! pos {
    () => ((file!(), line!()))
}

macro_rules! dump_and_die {
    ($($pos:expr),*) => ({
        // FIXME(#18285): we cannot include the current position because
        // the macro span takes over the last frame's file/line.
        //
        // You might also be wondering why a major platform,
        // i686-pc-windows-msvc, is located in here. Some of the saga can be
        // found on #62897, but the tl;dr; is that it appears that if the
        // standard library doesn't have debug information or frame pointers,
        // which it doesn't by default on the test builders, then the stack
        // walking routines in dbghelp will randomly terminate the stack trace
        // in libstd without going further. Presumably the addition of frame
        // pointers and/or debuginfo fixes this since tests always work with
        // nightly compilers (which have debuginfo). In general though this test
        // is replicated in rust-lang/backtrace-rs and has extensive coverage
        // there, even on i686-pc-windows-msvc. We do the best we can in
        // rust-lang/rust to test it as well, but sometimes we just gotta keep
        // landing PRs.
        if cfg!(any(target_os = "android",
                    all(target_os = "linux", target_arch = "arm"),
                    all(target_env = "msvc", target_arch = "x86"),
                    target_os = "freebsd",
                    target_os = "dragonfly",
                    target_os = "openbsd")) {
            // skip these platforms as this support isn't implemented yet.
        } else {
            dump_filelines(&[$($pos),*]);
            panic!();
        }
    })
}

// we can't use a function as it will alter the backtrace
macro_rules! check {
    ($counter:expr; $($pos:expr),*) => ({
        if *$counter == 0 {
            dump_and_die!($($pos),*)
        } else {
            *$counter -= 1;
        }
    })
}

type Pos = (&'static str, u32);

// this goes to stdout and each line has to be occurred
// in the following backtrace to stderr with a correct order.
fn dump_filelines(filelines: &[Pos]) {
    for &(file, line) in filelines.iter().rev() {
        // extract a basename
        let basename = file.split(&['/', '\\'][..]).last().unwrap();
        println!("{}:{}", basename, line);
    }
}

#[inline(never)]
fn inner(counter: &mut i32, main_pos: Pos, outer_pos: Pos) {
    check!(counter; main_pos, outer_pos);
    check!(counter; main_pos, outer_pos);
    let inner_pos = pos!(); aux::callback(|aux_pos| {
        check!(counter; main_pos, outer_pos, inner_pos, aux_pos);
    });
    let inner_pos = pos!(); aux::callback_inlined(|aux_pos| {
        check!(counter; main_pos, outer_pos, inner_pos, aux_pos);
    });
}

// We emit the wrong location for the caller here when inlined on MSVC
#[cfg_attr(not(target_env = "msvc"), inline(always))]
#[cfg_attr(target_env = "msvc", inline(never))]
fn inner_inlined(counter: &mut i32, main_pos: Pos, outer_pos: Pos) {
    check!(counter; main_pos, outer_pos);
    check!(counter; main_pos, outer_pos);

    // Again, disable inlining for MSVC.
    #[cfg_attr(not(target_env = "msvc"), inline(always))]
    #[cfg_attr(target_env = "msvc", inline(never))]
    fn inner_further_inlined(counter: &mut i32, main_pos: Pos, outer_pos: Pos, inner_pos: Pos) {
        check!(counter; main_pos, outer_pos, inner_pos);
    }
    inner_further_inlined(counter, main_pos, outer_pos, pos!());

    let inner_pos = pos!(); aux::callback(|aux_pos| {
        check!(counter; main_pos, outer_pos, inner_pos, aux_pos);
    });
    let inner_pos = pos!(); aux::callback_inlined(|aux_pos| {
        check!(counter; main_pos, outer_pos, inner_pos, aux_pos);
    });

    // this tests a distinction between two independent calls to the inlined function.
    // (un)fortunately, LLVM somehow merges two consecutive such calls into one node.
    inner_further_inlined(counter, main_pos, outer_pos, pos!());
}

#[inline(never)]
fn outer(mut counter: i32, main_pos: Pos) {
    inner(&mut counter, main_pos, pos!());
    inner_inlined(&mut counter, main_pos, pos!());
}

fn check_trace(output: &str, error: &str) -> Result<(), String> {
    // reverse the position list so we can start with the last item (which was the first line)
    let mut remaining: Vec<&str> = output.lines().map(|s| s.trim()).rev().collect();

    if !error.contains("stack backtrace") {
        return Err(format!("no backtrace found in stderr:\n{}", error))
    }
    for line in error.lines() {
        if !remaining.is_empty() && line.contains(remaining.last().unwrap()) {
            remaining.pop();
        }
    }
    if !remaining.is_empty() {
        return Err(format!("trace does not match position list\n\
            still need to find {:?}\n\n\
            --- stdout\n{}\n\
            --- stderr\n{}",
            remaining, output, error))
    }
    Ok(())
}

fn run_test(me: &str) {
    use std::str;
    use std::process::Command;

    let mut i = 0;
    let mut errors = Vec::new();
    loop {
        let out = Command::new(me)
                          .env("RUST_BACKTRACE", "full")
                          .arg(i.to_string()).output().unwrap();
        let output = str::from_utf8(&out.stdout).unwrap();
        let error = str::from_utf8(&out.stderr).unwrap();
        if out.status.success() {
            assert!(output.contains("done."), "bad output for successful run: {}", output);
            break;
        } else {
            if let Err(e) = check_trace(output, error) {
                errors.push(e);
            }
        }
        i += 1;
    }
    if errors.len() > 0 {
        for error in errors {
            println!("---------------------------------------");
            println!("{}", error);
        }

        panic!("found some errors");
    }
}

#[inline(never)]
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 {
        let case = args[1].parse().unwrap();
        eprintln!("test case {}", case);
        outer(case, pos!());
        println!("done.");
    } else {
        run_test(&args[0]);
    }
}
