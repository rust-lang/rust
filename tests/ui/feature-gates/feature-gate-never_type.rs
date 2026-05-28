// Test that ! errors when used in illegal positions with feature(never_type) disabled

mod ungated {
    //! Functions returning `!` directly (as in `-> !`) and function pointers doing the same are
    //! allowed with no gates.

    fn panic() -> ! {
        panic!();
    }

    fn takes_fn_ptr(x: fn() -> !) -> ! {
        x()
    }
}

mod gated {
    //! All other mentions of the type are gated.

    trait Foo {
        type Wub;
    }

    type Ma = (u32, !, i32); //~ ERROR type is experimental
    type Meeshka = Vec<!>; //~ ERROR type is experimental
    type Mow = &'static fn(!) -> !; //~ ERROR type is experimental
    type Skwoz = &'static mut !; //~ ERROR type is experimental
    type Meow = fn() -> Result<(), !>; //~ ERROR type is experimental

    impl Foo for Meeshka {
        type Wub = !; //~ ERROR type is experimental
    }

    fn look_ma_no_feature_gate<F: FnOnce() -> !>() {} //~ ERROR type is experimental

    fn tadam(f: &dyn Fn() -> !) {} //~ ERROR type is experimental

    fn toudoum() -> impl Fn() -> ! { //~ ERROR type is experimental
        || panic!()
    }

    fn infallible() -> Result<(), !> { //~ ERROR type is experimental
        Ok(())
    }
}

mod hack {
    //! There is a hack which, by exploiting the fact that `fn() -> !` can be named stably and that
    //! type system does not interact with stability, allows one to mention the never type while
    //! avoiding any and all feature gates. It is generally considered a "hack"/compiler bug, and
    //! thus users of this hack resign stability guarantees. However, fixing this is more trouble
    //! than good.

    trait F {
        type Ret;
    }

    impl<T> F for fn() -> T {
        type Ret = T;
    }

    type Never = <fn() -> ! as F>::Ret;

    fn damn(
        never: Never,
        _: &dyn Fn() -> Never,
    ) -> (impl Fn() -> Never, &'static mut Never, Never, u8) {
        (|| never, never, never, never)
    }
}

fn main() {}
