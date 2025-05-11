//@aux-build:proc_macro_attr.rs

#![warn(clippy::blocks_in_conditions)]
#![allow(
    unused,
    unnecessary_transmutes,
    clippy::needless_if,
    clippy::missing_transmute_annotations
)]
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
    //~^ nonminimal_bool
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

fn issue_9911() {
    if { return } {}

    let a = 1;
    if { if a == 1 { return } else { true } } {}
}

fn in_closure() {
    let v = vec![1, 2, 3];
    if v.into_iter()
        .filter(|x| {
            let y = x + 1;
            y > 3
        })
        .any(|x| x == 5)
    {
        println!("contains 4!");
    }
}

fn main() {}
