//@ revisions: current next
//@ [next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

// Regression test for <https://github.com/rust-lang/rust/issues/155092>.
// Used to crash

pub trait TraitItem {}
pub trait TraitInner {
    type Encoder;
}
trait Trait {
    type X<T>: TraitInner;
    fn a<T: TraitItem>() -> <Self::X<T> as TraitInner>::Encoder {
        todo!()
    }
}
fn a<T: Trait>() {
    T::a();
    //~^ ERROR type annotations needed
}

fn main() {}
