fn fn_pointer_static() -> usize {
    static FN: fn() -> usize = || 1;
    let res = FN() + 1;
    res
    //~^ ERROR: returning the result of a `let` binding from a block
    //~| NOTE: `-D clippy::let-and-return` implied by `-D warnings`
}

fn fn_pointer_const() -> usize {
    const FN: fn() -> usize = || 1;
    let res = FN() + 1;
    res
    //~^ ERROR: returning the result of a `let` binding from a block
}

fn deref_to_dyn_fn() -> usize {
    struct Derefs;
    impl std::ops::Deref for Derefs {
        type Target = dyn Fn() -> usize;

        fn deref(&self) -> &Self::Target {
            &|| 2
        }
    }
    static FN: Derefs = Derefs;
    let res = FN() + 1;
    res
    //~^ ERROR: returning the result of a `let` binding from a block
}

fn main() {}
