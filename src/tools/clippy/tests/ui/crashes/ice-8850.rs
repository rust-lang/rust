fn fn_pointer_static() -> usize {
    static FN: fn() -> usize = || 1;
    let res = FN() + 1;
    res
    //~^ let_and_return
}

fn fn_pointer_const() -> usize {
    const FN: fn() -> usize = || 1;
    let res = FN() + 1;
    res
    //~^ let_and_return
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
    //~^ let_and_return
}

fn main() {}
