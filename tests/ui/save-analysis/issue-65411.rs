// check-pass
// compile-flags: -Zsave-analysis

trait Trait { type Assoc; }
trait GenericTrait<T> {}
struct Wrapper<B> { b: B }

fn func() {
    // Processing associated path in impl block definition inside a function
    // body does not ICE
    impl<B: Trait> GenericTrait<B::Assoc> for Wrapper<B> {}
}


fn main() {}
