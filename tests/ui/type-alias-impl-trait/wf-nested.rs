// Well-formedness of nested opaque types, i.e. `impl Sized` in
// `type Outer = impl Trait<Assoc = impl Sized>`. We check that
// the nested type is well-formed, even though this would also
// be implied by the item bounds of the opaque being
// well-formed. See the comments below.
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

// We could theoretically allow this (and previously did), as even
// though the nested opaque is not well-formed, it can only be
// used by normalizing the projection
//    <OuterOpaque1<T> as Trait<&'static T>>::Out
// Assuming that we check that this projection is well-formed, the wf
// of the nested opaque is implied.
type OuterOpaque1<T> = impl Trait<&'static T, Out = impl Sized>;
#[define_opaque(OuterOpaque1)]
fn define<T>() -> OuterOpaque1<T> {}
//~^ ERROR `T` may not live long enough

fn define_rpit<T>() -> impl Trait<&'static T, Out = impl Sized> {}
//~^ ERROR the parameter type `T` may not live long enough

// Similar to `define` but here `impl Sized` can be referenced directly as
// InnerOpaque<T>, so the `'static` bound is definitely required for
// soundness.
type InnerOpaque<T> = impl Sized;
type OuterOpaque2<T> = impl Trait<&'static T, Out = InnerOpaque<T>>;
#[define_opaque(OuterOpaque2)]
fn define_nested_rpit<T>() -> OuterOpaque2<T> {}
//~^ ERROR the parameter type `T` may not live long enough

fn main() {}
