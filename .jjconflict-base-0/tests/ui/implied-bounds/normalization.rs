// Test that we get implied bounds from complex projections after normalization.

//@ check-pass

// implementations wil ensure that
// WF(<T as Combine<'a>>::Ty) implies T: 'a
trait Combine<'a> {
    type Ty;
}

impl<'a, T: 'a> Combine<'a> for Box<T> {
    type Ty = &'a T;
}

// ======= Wrappers ======

// normalizes to a projection
struct WrapA<T>(T);
impl<'a, T> Combine<'a> for WrapA<T>
where
    T: Combine<'a>,
{
    type Ty = T::Ty;
}

// <WrapB<T> as Combine<'a>>::Ty normalizes to a type variable ?X
// with constraint `<T as Combine<'a>>::Ty == ?X`
struct WrapB<T>(T);
impl<'a, X, T> Combine<'a> for WrapB<T>
where
    T: Combine<'a, Ty = X>,
{
    type Ty = X;
}

// <WrapC<T> as Combine<'a>>::Ty normalizes to `&'a &'?x ()`
// with constraint `<T as Combine<'a>>::Ty == &'a &'?x ()`
struct WrapC<T>(T);
impl<'a, 'x: 'a, T> Combine<'a> for WrapC<T>
where
    T: Combine<'a, Ty = &'a &'x ()>,
{
    type Ty = &'a &'x ();
}

//==== Test implied bounds ======

fn test_wrap<'a, 'b, 'c1, 'c2, A, B>(
    _: <WrapA<Box<A>> as Combine<'a>>::Ty,        // normalized: &'a A
    _: <WrapB<Box<B>> as Combine<'b>>::Ty,        // normalized: &'b B
    _: <WrapC<Box<&'c1 ()>> as Combine<'c2>>::Ty, // normalized: &'c2 &'c1 ()
) {
    None::<&'a A>;
    None::<&'b B>;
    None::<&'c2 &'c1 ()>;
}

fn main() {}
