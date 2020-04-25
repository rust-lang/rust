// run-rustfix

#![warn(clippy::implicit_return)]
#![allow(clippy::needless_return, unused)]

fn test_end_of_fn() -> bool {
    if true {
        // no error!
        return true;
    }

    true
}

#[allow(clippy::needless_bool)]
fn test_if_block() -> bool {
    if true {
        true
    } else {
        false
    }
}

#[rustfmt::skip]
fn test_match(x: bool) -> bool {
    match x {
        true => false,
        false => { true },
    }
}

#[allow(clippy::needless_return)]
fn test_match_with_unreachable(x: bool) -> bool {
    match x {
        true => return false,
        false => unreachable!(),
    }
}

#[allow(clippy::never_loop)]
fn test_loop() -> bool {
    loop {
        break true;
    }
}

#[allow(clippy::never_loop)]
fn test_loop_with_block() -> bool {
    loop {
        {
            break true;
        }
    }
}

#[allow(clippy::never_loop)]
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

fn main() {
    let _ = test_end_of_fn();
    let _ = test_if_block();
    let _ = test_match(true);
    let _ = test_match_with_unreachable(true);
    let _ = test_loop();
    let _ = test_loop_with_block();
    let _ = test_loop_with_nests();
    let _ = test_loop_with_if_let();
    test_closure();
    let _ = test_return_macro();
}
