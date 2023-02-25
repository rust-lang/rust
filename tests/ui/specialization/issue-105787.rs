// check-pass
// known-bug: #105787
trait ToUnit<'a> {
    type Unit;
}

struct LocalTy;
impl<'a> ToUnit<'a> for *const LocalTy {
    type Unit = ();
}

impl<'a, T: Copy + ?Sized> ToUnit<'a> for *const T {
    type Unit = ();
}

trait Overlap<T> {
    type Assoc;
}

type Assoc<'a, T> = <*const T as ToUnit<'a>>::Unit;

impl<T> Overlap<T> for T {
    type Assoc = usize;
}

impl<T> Overlap<for<'a> fn(&'a (), Assoc<'a, T>)> for T
where
    for<'a> *const T: ToUnit<'a>,
{
    type Assoc = Box<usize>;
}

fn foo<T: Overlap<U>, U>(x: T::Assoc) -> T::Assoc {
    x
}

fn main() {
    foo::<for<'a> fn(&'a (), ()), for<'a> fn(&'a (), ())>(3usize);
}
