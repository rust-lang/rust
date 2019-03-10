#![warn(clippy::block_in_if_condition_expr)]
#![warn(clippy::block_in_if_condition_stmt)]
#![allow(unused, clippy::let_and_return)]
#![warn(clippy::nonminimal_bool)]

macro_rules! blocky {
    () => {{
        true
    }};
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
        let x = 3;
        x == 3
    } {
        6
    } else {
        10
    }
}

fn condition_has_block_with_single_expression() -> i32 {
    if { true } {
        6
    } else {
        10
    }
}

fn predicate<F: FnOnce(T) -> bool, T>(pfn: F, val: T) -> bool {
    pfn(val)
}

fn pred_test() {
    let v = 3;
    let sky = "blue";
    // This is a sneaky case, where the block isn't directly in the condition,
    // but is actually nside a closure that the condition is using.
    // The same principle applies -- add some extra expressions to make sure
    // linter isn't confused by them.
    if v == 3
        && sky == "blue"
        && predicate(
            |x| {
                let target = 3;
                x == target
            },
            v,
        )
    {}

    if predicate(
        |x| {
            let target = 3;
            x == target
        },
        v,
    ) {}
}

fn condition_is_normal() -> i32 {
    let x = 3;
    if true && x == 3 {
        6
    } else {
        10
    }
}

fn closure_without_block() {
    if predicate(|x| x == 3, 6) {}
}

fn condition_is_unsafe_block() {
    let a: i32 = 1;

    // this should not warn because the condition is an unsafe block
    if unsafe { 1u32 == std::mem::transmute(a) } {
        println!("1u32 == a");
    }
}

fn main() {}

fn macro_in_closure() {
    let option = Some(true);

    if option.unwrap_or_else(|| unimplemented!()) {
        unimplemented!()
    }
}
