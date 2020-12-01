#![feature(unboxed_closures)]

trait Lt<'a> {
    type T;
}
impl<'a> Lt<'a> for () {
    type T = ();
}

fn main() {
    let v: <() as Lt<'_>>::T = ();
    let f: &mut dyn FnMut<(_,), Output = ()> = &mut |_: <() as Lt<'_>>::T| {};
    //~^ ERROR: the size for values of type `<() as Lt<'_>>::T` cannot be known
    f(v);
}
