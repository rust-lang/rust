// Test `?Sized` types not allowed in fields (except the last one).

struct S1<X: ?Sized> {
    f1: X,
    //~^ ERROR the size for values of type
    f2: isize,
}
struct S2<X: ?Sized> {
    f: isize,
    g: X,
    //~^ ERROR the size for values of type
    h: isize,
}
struct S3 {
    f: str,
    //~^ ERROR the size for values of type
    g: [usize]
}
struct S4 {
    f: [u8],
    //~^ ERROR the size for values of type
    g: usize
}
enum E<X: ?Sized> {
    V1(X, isize),
    //~^ ERROR the size for values of type
}
enum F<X: ?Sized> {
    V2{f1: X, f: isize},
    //~^ ERROR the size for values of type
}

pub fn main() {
}
