//@ run-fail
//@ error-pattern:meep
//@ needs-subprocess

fn f(_a: isize, _b: isize, _c: Box<isize>) {
    panic!("moop");
}

fn main() {
    f(1, panic!("meep"), Box::new(42));
}
