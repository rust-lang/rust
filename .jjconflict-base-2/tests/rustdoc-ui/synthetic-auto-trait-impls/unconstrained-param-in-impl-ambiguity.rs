// We used to ICE here while trying to synthesize auto trait impls.
// issue: 112828

struct Outer(Inner);
struct Inner;

unsafe impl<Q: Trait> Send for Inner {}
//~^ ERROR the type parameter `Q` is not constrained by the impl trait, self type, or predicates

trait Trait {}
