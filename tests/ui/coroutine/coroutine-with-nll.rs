#![feature(coroutines)]

fn main() {
    #[coroutine]
    || {
        // The reference in `_a` is a Legal with NLL since it ends before the yield
        let _a = &mut true;
        let b = &mut true;
        //~^ ERROR borrow may still be in use when coroutine yields
        yield ();
        println!("{}", b);
    };
}
