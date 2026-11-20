//! Regression test for <https://github.com/rust-lang/rust/issues/29668>.
//! Test functions can return unnameable types.
//@ run-pass

mod m1 {
    mod m2 {
        #[derive(Debug)]
        pub struct A;
    }
    use self::m2::A;
    pub fn x() -> A { A }
}

fn main() {
    let x = m1::x();
    println!("{:?}", x);
}
