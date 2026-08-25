//! Regression test for https://github.com/rust-lang/rust/issues/15129

//@ run-pass

pub enum T {
    T1(()),
    T2(()),
}

pub enum V {
    V1(isize),
    V2(bool),
}

fn foo(x: (T, V)) -> String {
    match x {
        (T::T1(()), V::V1(i)) => format!("T1(()), V1({})", i),
        (T::T2(()), V::V2(b)) => format!("T2(()), V2({})", b),
        _ => String::new(),
    }
}

fn main() {
    assert_eq!(foo((T::T1(()), V::V1(99))), "T1(()), V1(99)".to_string());
    assert_eq!(foo((T::T2(()), V::V2(true))), "T2(()), V2(true)".to_string());
}
