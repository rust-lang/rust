// Test `?Sized` local variables.

trait T {}

fn f1<W: ?Sized, X: ?Sized, Y: ?Sized, Z: ?Sized>(x: &X) {
    let _: W; // <-- this is OK, no bindings created, no initializer.
    let _: (isize, (X, isize));
    //~^ ERROR the size for values of type
    let y: Y;
    //~^ ERROR the size for values of type
    let y: (isize, (Z, usize));
    //~^ ERROR the size for values of type
}
fn f2<X: ?Sized, Y: ?Sized>(x: &X) {
    let y: X;
    //~^ ERROR the size for values of type
    let y: (isize, (Y, isize));
    //~^ ERROR the size for values of type
}

fn f3<X: ?Sized>(x1: Box<X>, x2: Box<X>, x3: Box<X>) {
    let y: X = *x1;
    //~^ ERROR the size for values of type
    let y = *x2;
    //~^ ERROR the size for values of type
    let (y, z) = (*x3, 4);
    //~^ ERROR the size for values of type
}
fn f4<X: ?Sized + T>(x1: Box<X>, x2: Box<X>, x3: Box<X>) {
    let y: X = *x1;
    //~^ ERROR the size for values of type
    let y = *x2;
    //~^ ERROR the size for values of type
    let (y, z) = (*x3, 4);
    //~^ ERROR the size for values of type
}

fn g1<X: ?Sized>(x: X) {}
//~^ ERROR the size for values of type
fn g2<X: ?Sized + T>(x: X) {}
//~^ ERROR the size for values of type

pub fn main() {
}
