fn f(arg: &mut A) {
    arg.a = 100;
}

#[derive(Copy, Clone)]
struct A { a: isize }

pub fn main() {
    let mut x = A {a: 10};
    f(&mut x);
    assert_eq!(x.a, 100);
    x.a = 20;
    let mut y = x;
    f(&mut y);
    assert_eq!(x.a, 20);
}
