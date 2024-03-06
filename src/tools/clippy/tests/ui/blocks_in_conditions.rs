//@aux-build:proc_macro_attr.rs

#![warn(clippy::blocks_in_conditions)]
#![allow(unused, clippy::let_and_return, clippy::needless_if)]
#![warn(clippy::nonminimal_bool)]

macro_rules! blocky {
    () => {{ true }};
}

macro_rules! blocky_too {
    () => {{
        let r = true;
        r
    }};
}

fn macro_if() {
    if blocky!() {}

    if blocky_too!() {}
}

fn condition_has_block() -> i32 {
    if {
        //~^ ERROR: in an `if` condition, avoid complex blocks or closures with blocks; instead, move the block or closure higher and bind it with a `let`
        let x = 3;
        x == 3
    } {
        6
    } else {
        10
    }
}

fn condition_has_block_with_single_expression() -> i32 {
    if { true } { 6 } else { 10 }
    //~^ ERROR: omit braces around single expression condition
}

fn condition_is_normal() -> i32 {
    let x = 3;
    if true && x == 3 { 6 } else { 10 }
}

fn condition_is_unsafe_block() {
    let a: i32 = 1;

    // this should not warn because the condition is an unsafe block
    if unsafe { 1u32 == std::mem::transmute(a) } {
        println!("1u32 == a");
    }
}

fn block_in_assert() {
    let opt = Some(42);
    assert!(
        opt.as_ref()
            .map(|val| {
                let mut v = val * 2;
                v -= 1;
                v * 3
            })
            .is_some()
    );
}

// issue #11814
fn block_in_match_expr(num: i32) -> i32 {
    match {
        //~^ ERROR: in a `match` scrutinee, avoid complex blocks or closures with blocks; instead, move the block or closure higher and bind it with a `let`
        let opt = Some(2);
        opt
    } {
        Some(0) => 1,
        Some(n) => num * 2,
        None => 0,
    };

    match unsafe {
        let hearty_hearty_hearty = vec![240, 159, 146, 150];
        String::from_utf8_unchecked(hearty_hearty_hearty).as_str()
    } {
        "💖" => 1,
        "what" => 2,
        _ => 3,
    }
}

// issue #12162
macro_rules! timed {
    ($name:expr, $body:expr $(,)?) => {{
        let __scope = ();
        $body
    }};
}

fn issue_12162() {
    if timed!("check this!", false) {
        println!();
    }
}

mod issue_12016 {
    #[proc_macro_attr::fake_desugar_await]
    pub async fn await_becomes_block() -> i32 {
        match Some(1).await {
            Some(1) => 2,
            Some(2) => 3,
            _ => 0,
        }
    }
}

fn main() {}
