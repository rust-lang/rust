// Regression test for #125325

// Tests that we suggest changing an `impl Fn` param
// to `impl FnMut` when the provided closure arg
// is trying to mutate the closure env.
// Ensures that it works that way for both
// functions and methods

struct S;

impl S {
    fn assoc_func(&self, _f: impl Fn()) -> usize {
        0
    }
}

fn func(_f: impl Fn()) -> usize {
    0
}

fn test_func(s: &S) -> usize {
    let mut x = ();
    s.assoc_func(|| x = ());
    //~^ ERROR cannot assign to `x`, as it is a captured variable in a `Fn` closure
    func(|| x = ())
    //~^ ERROR cannot assign to `x`, as it is a captured variable in a `Fn` closure
}

fn main() {}
