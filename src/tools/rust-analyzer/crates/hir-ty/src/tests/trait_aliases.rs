use crate::tests::check_types;

#[test]
fn projection() {
    check_types(
        r#"
#![feature(trait_alias)]

pub trait A {
    type Output;
}

pub trait B = A<Output = u32>;

pub fn a<T: B>(x: T::Output) {
    x;
//  ^ u32
}
"#,
    );
}
