// Edition 2024 lint for change in drop order at tail expression
// This lint is to capture potential change in program semantics
// due to implementation of RFC 3606 <https://github.com/rust-lang/rfcs/pull/3606>
//@ edition: 2021

#![deny(tail_expr_drop_order)] //~ NOTE: the lint level is defined here
#![allow(dropping_copy_types)]

struct LoudDropper;
impl Drop for LoudDropper {
    //~^ NOTE: `#1` invokes this custom destructor
    //~| NOTE: `x` invokes this custom destructor
    //~| NOTE: `#1` invokes this custom destructor
    //~| NOTE: `x` invokes this custom destructor
    //~| NOTE: `#1` invokes this custom destructor
    //~| NOTE: `x` invokes this custom destructor
    //~| NOTE: `#1` invokes this custom destructor
    //~| NOTE: `x` invokes this custom destructor
    //~| NOTE: `#1` invokes this custom destructor
    //~| NOTE: `_x` invokes this custom destructor
    //~| NOTE: `#1` invokes this custom destructor
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
    //~^ NOTE: `x` calls a custom destructor
    //~| NOTE: `x` will be dropped later as of Edition 2024
    // Should lint
    x.get() + LoudDropper.get()
    //~^ ERROR: relative drop order changing in Rust 2024
    //~| NOTE: this value will be stored in a temporary; let us call it `#1`
    //~| NOTE: up until Edition 2021 `#1` is dropped last but will be dropped earlier in Edition 2024
    //~| WARN: this changes meaning in Rust 2024
    //~| NOTE: most of the time, changing drop order is harmless; inspect the `impl Drop`s for side effects
    //~| NOTE: for more information, see
}
//~^ NOTE: now the temporary value is dropped here, before the local variables in the block or statement

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
        let x = LoudDropper;
        //~^ NOTE: `x` calls a custom destructor
        //~| NOTE: `x` will be dropped later as of Edition 2024
        // Should lint
        x.get() + LoudDropper.get()
        //~^ ERROR: relative drop order changing in Rust 2024
        //~| NOTE: this value will be stored in a temporary; let us call it `#1`
        //~| NOTE: up until Edition 2021 `#1` is dropped last but will be dropped earlier in Edition 2024
        //~| WARN: this changes meaning in Rust 2024
        //~| NOTE: most of the time, changing drop order is harmless; inspect the `impl Drop`s for side effects
        //~| NOTE: for more information, see
    }
    //~^ NOTE: now the temporary value is dropped here, before the local variables in the block or statement
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
    let x = LoudDropper;
    //~^ NOTE: `x` calls a custom destructor
    //~| NOTE: `x` will be dropped later as of Edition 2024
    { LoudDropper.get() }
    //~^ ERROR: relative drop order changing in Rust 2024
    //~| NOTE: this value will be stored in a temporary; let us call it `#1`
    //~| NOTE: up until Edition 2021 `#1` is dropped last but will be dropped earlier in Edition 2024
    //~| WARN: this changes meaning in Rust 2024
    //~| NOTE: most of the time, changing drop order is harmless; inspect the `impl Drop`s for side effects
    //~| NOTE: for more information, see
}
//~^ NOTE: now the temporary value is dropped here, before the local variables in the block or statement

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
    async fn f() {
        async fn f() {}
        let x = LoudDropper;
        f().await;
        drop(x);
    }

    let future = f();
    //~^ NOTE: `future` calls a custom destructor
    //~| NOTE: `future` will be dropped later as of Edition 2024
    LoudDropper.get()
    //~^ ERROR: relative drop order changing in Rust 2024
    //~| WARN: this changes meaning in Rust 2024
    //~| NOTE: this value will be stored in a temporary; let us call it `#1`
    //~| NOTE: up until Edition 2021 `#1` is dropped last but will be dropped earlier in Edition 2024
    //~| NOTE: most of the time, changing drop order is harmless; inspect the `impl Drop`s for side effects
    //~| NOTE: for more information, see
}
//~^ NOTE: now the temporary value is dropped here, before the local variables in the block or statement

fn should_lint_generics<T: Default>() -> &'static str {
    fn extract<T>(_: &T) -> &'static str {
        todo!()
    }
    let x = T::default();
    //~^ NOTE: `x` calls a custom destructor
    //~| NOTE: `x` will be dropped later as of Edition 2024
    extract(&T::default())
    //~^ ERROR: relative drop order changing in Rust 2024
    //~| WARN: this changes meaning in Rust 2024
    //~| NOTE: this value will be stored in a temporary; let us call it `#1`
    //~| NOTE: up until Edition 2021 `#1` is dropped last but will be dropped earlier in Edition 2024
    //~| NOTE: most of the time, changing drop order is harmless; inspect the `impl Drop`s for side effects
    //~| NOTE: for more information, see
}
//~^ NOTE: now the temporary value is dropped here, before the local variables in the block or statement

fn should_lint_adt() -> i32 {
    let x: Result<LoudDropper, ()> = Ok(LoudDropper);
    //~^ NOTE: `x` calls a custom destructor
    //~| NOTE: `x` will be dropped later as of Edition 2024
    LoudDropper.get()
    //~^ ERROR: relative drop order changing in Rust 2024
    //~| WARN: this changes meaning in Rust 2024
    //~| NOTE: this value will be stored in a temporary; let us call it `#1`
    //~| NOTE: up until Edition 2021 `#1` is dropped last but will be dropped earlier in Edition 2024
    //~| NOTE: most of the time, changing drop order is harmless; inspect the `impl Drop`s for side effects
    //~| NOTE: for more information, see
}
//~^ NOTE: now the temporary value is dropped here, before the local variables in the block or statement

fn should_not_lint_insign_dtor() -> i32 {
    let x = String::new();
    LoudDropper.get()
}

fn should_lint_with_dtor_span() -> i32 {
    struct LoudDropper3;
    impl Drop for LoudDropper3 {
        //~^ NOTE: `#1` invokes this custom destructor
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
        //~^ NOTE: `x` invokes this custom destructor
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
    //~^ NOTE: `x` calls a custom destructor
    //~| NOTE: `x` will be dropped later as of Edition 2024
    LoudDropper3.get()
    //~^ ERROR: relative drop order changing in Rust 2024
    //~| NOTE: this value will be stored in a temporary; let us call it `#1`
    //~| NOTE: up until Edition 2021 `#1` is dropped last but will be dropped earlier in Edition 2024
    //~| WARN: this changes meaning in Rust 2024
    //~| NOTE: most of the time, changing drop order is harmless; inspect the `impl Drop`s for side effects
    //~| NOTE: for more information, see
}
//~^ NOTE: now the temporary value is dropped here, before the local variables in the block or statement

fn should_lint_with_transient_drops() {
    drop((
        {
            LoudDropper.get()
            //~^ ERROR: relative drop order changing in Rust 2024
            //~| NOTE: this value will be stored in a temporary; let us call it `#1`
            //~| NOTE: up until Edition 2021 `#1` is dropped last but will be dropped earlier in Edition 2024
            //~| WARN: this changes meaning in Rust 2024
            //~| NOTE: most of the time, changing drop order is harmless; inspect the `impl Drop`s for side effects
            //~| NOTE: for more information, see
        },
        {
            let _x = LoudDropper;
            //~^ NOTE: `_x` calls a custom destructor
            //~| NOTE: `_x` will be dropped later as of Edition 2024
        },
    ));
    //~^ NOTE: now the temporary value is dropped here, before the local variables in the block or statement
}

fn main() {}
