// The suggestion has to carry the trait's `use<..>` list over, expressed in terms of the impl's own
// generics, or applying it turns this warning into a hard error.

#![deny(refining_impl_trait)]

trait Captures {
    fn f(&self) -> impl use<Self> + Send;
}

impl Captures for () {
    fn f(&self) -> () {}
    //~^ ERROR impl trait in impl method signature does not match trait method signature
}

struct W<T>(T);

impl<T: Send> Captures for W<T> {
    fn f(&self) -> () {}
    //~^ ERROR impl trait in impl method signature does not match trait method signature
}

trait CapturesParam<T> {
    fn f(&self) -> impl use<Self, T> + Send;
}

impl CapturesParam<u8> for () {
    fn f(&self) -> () {}
    //~^ ERROR impl trait in impl method signature does not match trait method signature
}

// The trait captures everything in scope, so the suggestion is already correct.
trait NoCaptures {
    fn f(&self) -> impl Send;
}

impl NoCaptures for () {
    fn f(&self) -> () {}
    //~^ ERROR impl trait in impl method signature does not match trait method signature
}

// The impl would capture the same lifetime by default, so the suggestion is already correct.
trait CapturesLifetime {
    fn f<'a>(&'a self) -> impl use<'a, Self> + Send;
}

impl CapturesLifetime for () {
    fn f<'a>(&'a self) -> () {}
    //~^ ERROR impl trait in impl method signature does not match trait method signature
}

fn main() {}
