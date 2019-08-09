#![feature(generators)]

fn main() {
    || {
        // The reference in `_a` is a Legal with NLL since it ends before the yield
        let _a = &mut true;
        let b = &mut true;
        //~^ borrow may still be in use when generator yields
        yield ();
        println!("{}", b);
    };
}
