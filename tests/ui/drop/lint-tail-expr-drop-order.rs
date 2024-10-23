// Edition 2024 lint for change in drop order at tail expression
// This lint is to capture potential change in program semantics
// due to implementation of RFC 3606 <https://github.com/rust-lang/rfcs/pull/3606>
//@ edition: 2021
//@ build-fail

#![deny(tail_expr_drop_order)] //~ NOTE: the lint level is defined here
#![feature(shorter_tail_lifetimes)]

struct LoudDropper;
impl Drop for LoudDropper {
    //~^ NOTE: dropping the temporary runs this custom `Drop` impl, which will run first in Rust 2024
    //~| NOTE: dropping the local runs this custom `Drop` impl, which will run second in Rust 2024
    //~| NOTE: dropping the temporary runs this custom `Drop` impl, which will run first in Rust 2024
    //~| NOTE: dropping the local runs this custom `Drop` impl, which will run second in Rust 2024
    //~| NOTE: dropping the temporary runs this custom `Drop` impl, which will run first in Rust 2024
    //~| NOTE: dropping the local runs this custom `Drop` impl, which will run second in Rust 2024
    //~| NOTE: dropping the temporary runs this custom `Drop` impl, which will run first in Rust 2024
    //~| NOTE: dropping the local runs this custom `Drop` impl, which will run second in Rust 2024
    //~| NOTE: dropping the temporary runs this custom `Drop` impl, which will run first in Rust 2024
    //~| NOTE: dropping the local runs this custom `Drop` impl, which will run second in Rust 2024
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
    //~^ NOTE: temporary will be dropped on exiting the block, before the block's local variables
    let x = LoudDropper;
    //~^ NOTE: in Rust 2024, this local variable or temporary value will be dropped second
    // Should lint
    x.get() + LoudDropper.get()
    //~^ ERROR: relative drop order changing in Rust 2024
    //~| NOTE: in Rust 2024, this temporary will be dropped first
    //~| WARN: this changes meaning in Rust 2024
    //~| NOTE: most of the time, changing drop order is harmless; inspect the `impl Drop`s for side effects
    //~| NOTE: for more information, see issue #123739
}

fn should_not_lint_closure() -> impl FnOnce() -> i32 {
    let x = LoudDropper;
    move || {
        // Should not lint because ...
        x.get() + LoudDropper.get()
    }
    // ^ closure captures like `x` are always dropped last by contract
}

fn should_lint_in_nested_items() {
    fn should_lint_me() -> i32 {
        //~^ NOTE: temporary will be dropped on exiting the block, before the block's local variables
        let x = LoudDropper;
        //~^ NOTE: in Rust 2024, this local variable or temporary value will be dropped second
        // Should lint
        x.get() + LoudDropper.get()
        //~^ ERROR: relative drop order changing in Rust 2024
        //~| NOTE: in Rust 2024, this temporary will be dropped first
        //~| WARN: this changes meaning in Rust 2024
        //~| NOTE: most of the time, changing drop order is harmless; inspect the `impl Drop`s for side effects
        //~| NOTE: for more information, see issue #123739
    }
}

fn should_not_lint_params(x: LoudDropper) -> i32 {
    // Should not lint because ...
    x.get() + LoudDropper.get()
}
// ^ function parameters like `x` are always dropped last

fn should_not_lint() -> i32 {
    let x = LoudDropper;
    // Should not lint
    x.get()
}

fn should_lint_in_nested_block() -> i32 {
    //~^ NOTE: temporary will be dropped on exiting the block, before the block's local variables
    let x = LoudDropper;
    //~^ NOTE: in Rust 2024, this local variable or temporary value will be dropped second
    { LoudDropper.get() }
    //~^ ERROR: relative drop order changing in Rust 2024
    //~| NOTE: in Rust 2024, this temporary will be dropped first
    //~| WARN: this changes meaning in Rust 2024
    //~| NOTE: most of the time, changing drop order is harmless; inspect the `impl Drop`s for side effects
    //~| NOTE: for more information, see issue #123739
}

fn should_not_lint_in_match_arm() -> i32 {
    let x = LoudDropper;
    // Should not lint because Edition 2021 drops temporaries in blocks earlier already
    match &x {
        _ => LoudDropper.get(),
    }
}

fn should_not_lint_when_consumed() -> (LoudDropper, i32) {
    let x = LoudDropper;
    // Should not lint because `LoudDropper` is consumed by the return value
    (LoudDropper, x.get())
}

struct MyAdt {
    a: LoudDropper,
    b: LoudDropper,
}

fn should_not_lint_when_consumed_in_ctor() -> MyAdt {
    let a = LoudDropper;
    // Should not lint
    MyAdt { a, b: LoudDropper }
}

fn should_not_lint_when_moved() -> i32 {
    let x = LoudDropper;
    drop(x);
    // Should not lint because `x` is not live
    LoudDropper.get()
}

fn should_lint_into_async_body() -> i32 {
    //~^ NOTE: temporary will be dropped on exiting the block, before the block's local variables
    async fn f() {
        async fn f() {}
        let x = LoudDropper;
        f().await;
        drop(x);
    }

    let future = f();
    //~^ NOTE: in Rust 2024, this local variable or temporary value will be dropped second
    LoudDropper.get()
    //~^ ERROR: relative drop order changing in Rust 2024
    //~| WARN: this changes meaning in Rust 2024
    //~| NOTE: in Rust 2024, this temporary will be dropped first
    //~| NOTE: most of the time, changing drop order is harmless; inspect the `impl Drop`s for side effects
    //~| NOTE: for more information, see issue #123739
}

fn should_lint_generics<T: Default>() -> &'static str {
    //~^ NOTE: temporary will be dropped on exiting the block, before the block's local variables
    fn extract<T>(_: &T) -> &'static str {
        todo!()
    }
    let x = T::default();
    //~^ NOTE: in Rust 2024, this local variable or temporary value will be dropped second
    extract(&T::default())
    //~^ ERROR: relative drop order changing in Rust 2024
    //~| WARN: this changes meaning in Rust 2024
    //~| NOTE: in Rust 2024, this temporary will be dropped first
    //~| NOTE: most of the time, changing drop order is harmless; inspect the `impl Drop`s for side effects
    //~| NOTE: for more information, see issue #123739
}

fn should_lint_adt() -> i32 {
    //~^ NOTE: temporary will be dropped on exiting the block, before the block's local variables
    let x: Result<LoudDropper, ()> = Ok(LoudDropper);
    //~^ NOTE: in Rust 2024, this local variable or temporary value will be dropped second
    LoudDropper.get()
    //~^ ERROR: relative drop order changing in Rust 2024
    //~| WARN: this changes meaning in Rust 2024
    //~| NOTE: in Rust 2024, this temporary will be dropped first
    //~| NOTE: most of the time, changing drop order is harmless; inspect the `impl Drop`s for side effects
    //~| NOTE: for more information, see issue #123739
}

fn should_not_lint_insign_dtor() -> i32 {
    let x = String::new();
    LoudDropper.get()
}

fn should_lint_with_dtor_span() -> i32 {
    //~^ NOTE: temporary will be dropped on exiting the block, before the block's local variables
    struct LoudDropper3;
    impl Drop for LoudDropper3 {
        //~^ NOTE: dropping the temporary runs this custom `Drop` impl, which will run first in Rust 2024
        fn drop(&mut self) {
            println!("loud drop");
        }
    }
    impl LoudDropper3 {
        fn get(&self) -> i32 {
            0
        }
    }
    struct LoudDropper2;
    impl Drop for LoudDropper2 {
        //~^ NOTE: dropping the local runs this custom `Drop` impl, which will run second in Rust 2024
        fn drop(&mut self) {
            println!("loud drop");
        }
    }
    impl LoudDropper2 {
        fn get(&self) -> i32 {
            0
        }
    }

    let x = LoudDropper2;
    //~^ NOTE: in Rust 2024, this local variable or temporary value will be dropped second
    LoudDropper3.get()
    //~^ ERROR: relative drop order changing in Rust 2024
    //~| NOTE: in Rust 2024, this temporary will be dropped first
    //~| WARN: this changes meaning in Rust 2024
    //~| NOTE: most of the time, changing drop order is harmless; inspect the `impl Drop`s for side effects
    //~| NOTE: for more information, see issue #123739
}

fn main() {}
