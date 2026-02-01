trait Outer { type Ty<'a, T: 'a + ?Sized>; }
trait Inner {}

fn f<'r, T: Outer>(x: T::Ty<'r, dyn Inner + 'r>) { /*check*/ g::<T>(x) }
// FIXME: Ideally, we would deduce `dyn Inner + 'r` from bound `'a` on ty param `T` of assoc ty `Ty`
//        but for that we'd need to somehow obtain the resolution of the type-relative path `T::Ty`
//        from HIR ty lowering (it resolves to `<T as Outer>::Ty`).
fn g<'r, T: Outer>(x: T::Ty<'r, dyn Inner>) {}
//~^ ERROR lifetime bound for this object type cannot be deduced from context

fn main() {}
