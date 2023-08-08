// Well-formedness of nested opaque types, i.e. `impl Sized` in
// `type Outer = impl Trait<Assoc = impl Sized>`.
// See the comments below.
//
// revisions: pass pass_sound fail
// [pass] check-pass
// [pass_sound] check-fail
// [fail] check-fail

#![feature(type_alias_impl_trait)]

struct IsStatic<T: 'static>(T);

trait Trait<In> {
    type Out;

    fn get(&self) -> Result<Self::Out, ()> {
        Err(())
    }
}

impl<T> Trait<&'static T> for () {
    type Out = IsStatic<T>;
}

// The hidden type for `impl Sized` is `IsStatic<T>`, which requires `T: 'static`.
// We know it is well-formed because it can *only* be referenced as a projection:
// <OuterOpaque<T> as Trait<&'static T>>::Out`.
// So any instantiation of the type already requires proving `T: 'static`.
#[cfg(pass)]
mod pass {
    use super::*;
    type OuterOpaque<T> = impl Trait<&'static T, Out = impl Sized>;
    fn define<T>() -> OuterOpaque<T> {}
}

// Test the soundness of `pass` - We should require `T: 'static` at the use site.
#[cfg(pass_sound)]
mod pass_sound {
    use super::*;
    type OuterOpaque<T> = impl Trait<&'static T, Out = impl Sized>;
    fn define<T>() -> OuterOpaque<T> {}

    fn test<T>() {
        let outer = define::<T>();
        let _ = outer.get(); //[pass_sound]~ ERROR `T` may not live long enough
    }
}

// Similar to `pass` but here `impl Sized` can be referenced directly as
// InnerOpaque<T>, so we require an explicit bound `T: 'static`.
#[cfg(fail)]
mod fail {
    use super::*;
    type InnerOpaque<T> = impl Sized; //[fail]~ ERROR `T` may not live long enough
    type OuterOpaque<T> = impl Trait<&'static T, Out = InnerOpaque<T>>;
    fn define<T>() -> OuterOpaque<T> {}
}

fn main() {}
