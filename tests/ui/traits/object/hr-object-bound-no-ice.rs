//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Regression test for <https://github.com/rust-lang/rust/issues/166470>

trait Super {
    type SuperAssoc;
}

trait Trait<'a>: Super<SuperAssoc = for<'b> fn(&'a <Self as Trait<'b>>::Assoc)> {
    type Assoc;
    fn call(&self);
}

fn boom<'c, T>(t: &'c dyn Trait<'static, Assoc = T, SuperAssoc = fn(&'static T)>) {
    // We used to ICE while trying to prove
    // `dyn Trait<'static, Assoc = T, SuperAssoc = T>: Trait<'?x>`
    // here with object bound candidates.
    // We cannot replace projection `for<'^b> <Self as Trait<'^b>::Assoc`
    // from `Trait`'s super bound with object bound `<Self as Trait<'static>>::Assoc = T`
    // because `'^b` is a HR region and trying to equate it with `'static` caused the ICE.
    t.call();
    //[current]~^ ERROR: mismatched types
    //[current]~| ERROR: mismatched types
    //[next]~^^^ ERROR: higher-ranked lifetime error
    //[next]~| ERROR: higher-ranked lifetime error
}

fn main() {}
