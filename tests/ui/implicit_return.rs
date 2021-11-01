// run-rustfix

#![warn(clippy::implicit_return)]
#![allow(clippy::needless_return, clippy::needless_bool, unused, clippy::never_loop)]

fn test_end_of_fn() -> bool {
    if true {
        // no error!
        return true;
    }

    true
}

fn test_if_block() -> bool {
    if true { true } else { false }
}

#[rustfmt::skip]
fn test_match(x: bool) -> bool {
    match x {
        true => false,
        false => { true },
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
    }
}

fn test_loop_with_block() -> bool {
    loop {
        {
            break true;
        }
    }
}

fn test_loop_with_nests() -> bool {
    loop {
        if true {
            break true;
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
    let _ = || true;
}

fn test_panic() -> bool {
    panic!()
}

fn test_return_macro() -> String {
    format!("test {}", "test")
}

fn macro_branch_test() -> bool {
    macro_rules! m {
        ($t:expr, $f:expr) => {
            if true { $t } else { $f }
        };
    }
    m!(true, false)
}

fn loop_test() -> bool {
    'outer: loop {
        if true {
            break true;
        }

        let _ = loop {
            if false {
                break 'outer false;
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

fn divergent_test() -> bool {
    fn diverge() -> ! {
        panic!()
    }
    diverge()
}

// issue #6940
async fn foo() -> bool {
    true
}

fn main() {}
