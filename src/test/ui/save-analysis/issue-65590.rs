// check-pass
// compile-flags: -Zsave-analysis
// edition:2018

// Async desugaring for return types in (associated) functions introduces a
// separate definition internally, which we need to take into account
// (or else we ICE).
trait Trait { type Assoc; }
struct Struct;

async fn foobar<T: Trait>() -> T::Assoc {
    unimplemented!()
}

impl Struct {
    async fn foo<T: Trait>(&self) -> T::Assoc {
        unimplemented!()
    }
}

fn main() {}
