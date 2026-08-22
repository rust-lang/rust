// Regression test for <https://github.com/rust-lang/rust/issues/155538>.
//
// This must be a full build test: `check-pass` only emits metadata and
// therefore does not run the post-mono MIR validator that this ICEs in.
//@ build-pass

trait Apply {
    type Output<T: Trait>: Trait;
}
struct Identity;
impl Apply for Identity {
    type Output<T: Trait> = T;
}

struct Thing<A: Apply>(A);

trait Trait {}

impl<A: Apply> Trait for Thing<A> where <A as Apply>::Output<Self>: Trait {}

fn weird<A: Apply>(x: A) -> impl Trait {
    Thing(x)
}

fn main() {
    let _ = Box::new(weird(Identity)) as Box<dyn Trait>;
}
