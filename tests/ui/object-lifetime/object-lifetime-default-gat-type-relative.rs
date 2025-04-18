// FIXME: Explainer

trait Outer { type Ty<'a, T: 'a + ?Sized>; }
trait Inner {}

fn f<'r, T: Outer>(x: T::Ty<'r, dyn Inner>) {}
//~^ ERROR lifetime bound for this object type cannot be deduced from context

fn main() {}
