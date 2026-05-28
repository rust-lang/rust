//@ run-pass

fn f() -> isize {
    mod m {
        pub fn g() -> isize { 720 }
    }

    m::g()
}

pub fn main() {
    assert_eq!(f(), 720);
}
