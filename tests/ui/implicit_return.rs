//@aux-build: proc_macros.rs

#![warn(clippy::implicit_return)]
#![allow(clippy::needless_return, clippy::needless_bool, unused, clippy::never_loop)]

extern crate proc_macros;
use proc_macros::with_span;

fn test_end_of_fn() -> bool {
    if true {
        // no error!
        return true;
    }

    true
    //~^ implicit_return
}

fn test_if_block() -> bool {
    if true { true } else { false }
    //~^ implicit_return
    //~| implicit_return
}

#[rustfmt::skip]
fn test_match(x: bool) -> bool {
    match x {
        true => false,
        //~^ implicit_return
        false => { true },
        //~^ implicit_return
    }
}

fn test_match_with_unreachable(x: bool) -> bool {
    match x {
        true => return false,
        false => unreachable!(),
    }
}

fn test_loop() -> bool {
    loop {
        break true;
        //~^ implicit_return
    }
}

fn test_loop_with_block() -> bool {
    loop {
        {
            break true;
            //~^ implicit_return
        }
    }
}

fn test_loop_with_nests() -> bool {
    loop {
        if true {
            break true;
            //~^ implicit_return
        } else {
            let _ = true;
        }
    }
}

#[allow(clippy::redundant_pattern_matching)]
fn test_loop_with_if_let() -> bool {
    loop {
        if let Some(x) = Some(true) {
            return x;
        }
    }
}

fn test_closure() {
    #[rustfmt::skip]
    let _ = || { true };
    //~^ implicit_return
    let _ = || true;
    //~^ implicit_return
}

fn test_panic() -> bool {
    panic!()
}

fn test_return_macro() -> String {
    format!("test {}", "test")
    //~^ implicit_return
}

fn macro_branch_test() -> bool {
    macro_rules! m {
        ($t:expr, $f:expr) => {
            if true { $t } else { $f }
        };
    }
    m!(true, false)
    //~^ implicit_return
}

fn loop_test() -> bool {
    'outer: loop {
        if true {
            break true;
            //~^ implicit_return
        }

        let _ = loop {
            if false {
                break 'outer false;
                //~^ implicit_return
            }
            if true {
                break true;
            }
        };
    }
}

fn loop_macro_test() -> bool {
    macro_rules! m {
        ($e:expr) => {
            break $e
        };
    }
    loop {
        m!(true);
    }
}
//~^^^^ implicit_return

fn divergent_test() -> bool {
    fn diverge() -> ! {
        panic!()
    }
    diverge()
}

// issue #6940
async fn foo() -> bool {
    true
    //~^ implicit_return
}

fn main() {}

fn check_expect() -> bool {
    if true {
        // no error!
        return true;
    }

    #[expect(clippy::implicit_return)]
    true
}

with_span!(
    span

    fn dont_lint_proc_macro(x: usize) -> usize{
        x
    }
);

fn desugared_closure_14446() {
    let _ = async || 0;
    //~^ implicit_return
    #[rustfmt::skip]
    let _ = async || -> i32 { 0 };
    //~^ implicit_return
    let _ = async |a: i32| a;
    //~^ implicit_return
    #[rustfmt::skip]
    let _ = async |a: i32| { a };
    //~^ implicit_return

    let _ = async || return 0;
    let _ = async || -> i32 { return 0 };
    let _ = async |a: i32| return a;
    #[rustfmt::skip]
    let _ = async |a: i32| { return a; };

    let _ = async || foo().await;
    //~^ implicit_return
    let _ = async || {
        foo().await;
        foo().await
    };
    //~^^ implicit_return
    #[rustfmt::skip]
    let _ = async || { foo().await };
    //~^ implicit_return
    let _ = async || -> bool { foo().await };
    //~^ implicit_return

    let _ = async || return foo().await;
    let _ = async || {
        foo().await;
        return foo().await;
    };
    #[rustfmt::skip]
    let _ = async || { return foo().await; };
    let _ = async || -> bool {
        return foo().await;
    };
}
