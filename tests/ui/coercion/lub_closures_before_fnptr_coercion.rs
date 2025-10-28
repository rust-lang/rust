//@ check-pass
//@ compile-flags: -Znext-solver

#![feature(type_alias_impl_trait)]

// Test that when lubbing two equal closure tys with different
// structural identities (i.e. `PartialEq::eq` on `ty::Ty` would be false)
// we don't coerce-lub to a fnptr.
//
// Most of this test is involved jank to be able to leak the hidden type
// of an opaque with a hidden type of `Closure<C1, C2>`. This then allows
// us to substitute `C1` and `C2` for arbitrary types in the parent scope.
//
// See: <https://github.com/lcnr/random-rust-snippets/issues/13>

struct WaddupGamers<T, U>(Option<T>, U);
impl<T: Leak<Unpin = U>, U> Unpin for WaddupGamers<T, U> {}
unsafe impl<T: Leak<Send = U>, U> Send for WaddupGamers<T, U> {}
pub trait Leak {
    type Unpin;
    type Send;
}
impl<T> Leak for (T,) {
    type Unpin = T;
    type Send = T;
}
fn define<C1, C2>() -> impl Sized {
    WaddupGamers(None::<C1>, || ())
}

fn require_unpin<T: Unpin>(_: T) {}
fn require_send<T: Send>(_: T) {}
fn mk<T>() -> T { todo!() }

type NameMe<T> = impl Sized;
type NameMe2<T> = impl Sized;

#[define_opaque(NameMe, NameMe2)]
fn leak<T>()
where
    T: Leak<Unpin = NameMe<T>, Send = NameMe2<T>>,
{
    require_unpin(define::<T, for<'a> fn(&'a ())>());
    require_send(define::<T, for<'a> fn(&'a ())>());

    // This is the actual logic for lubbing two closures
    // with syntactically different `ty::Ty`s:

    // lhs: Closure<C1=T, C2=for<'a1> fn(&'a1 ())>
    let lhs = mk::<NameMe<T>>();
    // lhs: Closure<C1=T, C2=for<'a2> fn(&'a2 ())>
    let rhs = mk::<NameMe2<T>>();

    macro_rules! lub {
        ($lhs:expr, $rhs:expr) => {
            if true { $lhs } else { $rhs }
        };
    }

    // Lubbed to either:
    // - `Closure<C1=T, C2=for<'a> fn(&'a ())>`
    // - `fn(&())`
    let lubbed = lub!(lhs, rhs);

    // Use transmute to assert the size of `lubbed` is (), i.e.
    // that it is a ZST closure type not a fnptr.
    unsafe { std::mem::transmute::<_, ()>(lubbed) };
}

fn main() {}
