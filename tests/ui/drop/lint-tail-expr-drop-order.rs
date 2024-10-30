//@ edition: 2021

// Edition 2024 lint for change in drop order at tail expression
// This lint is to capture potential change in program semantics
// due to implementation of RFC 3606 <https://github.com/rust-lang/rfcs/pull/3606>

#![deny(tail_expr_drop_order)]

struct LoudDropper;
impl Drop for LoudDropper {
    fn drop(&mut self) {
        // This destructor should be considered significant because it is a custom destructor
        // and we will assume that the destructor can generate side effects arbitrarily so that
        // a change in drop order is visible.
        println!("loud drop");
    }
}
impl LoudDropper {
    fn get(&self) -> i32 {
        0
    }
}

fn should_lint() -> i32 {
    let x = LoudDropper;
    // Should lint
    x.get() + LoudDropper.get()
    //~^ ERROR: these values and local bindings have significant drop implementation that will have a different drop order from that of Edition 2021
    //~| WARN: this changes meaning in Rust 2024
}

fn should_lint_closure() -> impl FnOnce() -> i32 {
    let x = LoudDropper;
    move || x.get() + LoudDropper.get()
    //~^ ERROR: these values and local bindings have significant drop implementation that will have a different drop order from that of Edition 2021
    //~| WARN: this changes meaning in Rust 2024
}

fn should_not_lint() -> i32 {
    let x = LoudDropper;
    // Should not lint
    x.get()
}

fn should_not_lint_in_nested_block() -> i32 {
    let x = LoudDropper;
    // Should not lint because Edition 2021 drops temporaries in blocks earlier already
    { LoudDropper.get() }
}

fn should_not_lint_in_match_arm() -> i32 {
    let x = LoudDropper;
    // Should not lint because Edition 2021 drops temporaries in blocks earlier already
    match &x {
        _ => LoudDropper.get(),
    }
}

fn should_lint_in_nested_items() {
    fn should_lint_me() -> i32 {
        let x = LoudDropper;
        // Should lint
        x.get() + LoudDropper.get()
        //~^ ERROR: these values and local bindings have significant drop implementation that will have a different drop order from that of Edition 2021
        //~| WARN: this changes meaning in Rust 2024
    }
}

fn main() {}
