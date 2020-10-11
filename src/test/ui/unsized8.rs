// Test `?Sized` local variables.

trait T {}

fn f1<W: ?Sized, X: ?Sized, Y: ?Sized, Z: ?Sized>(x: &X) {
    let y: (isize, (Z, usize));
    //~^ ERROR the size for values of type
}
fn f2<X: ?Sized, Y: ?Sized>(x: &X) {
    let y: (isize, (Y, isize));
    //~^ ERROR the size for values of type
}

fn f3<X: ?Sized>(x1: Box<X>, x2: Box<X>, x3: Box<X>) {
    let (y, z) = (*x3, 4);
    //~^ ERROR the size for values of type
}
fn f4<X: ?Sized + T>(x1: Box<X>, x2: Box<X>, x3: Box<X>) {
    let (y, z) = (*x3, 4);
    //~^ ERROR the size for values of type
}

pub fn main() {
}
