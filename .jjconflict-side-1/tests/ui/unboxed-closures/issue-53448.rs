//@ check-pass

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
    f(v);
}
