#![crate_name = "foo"]

pub trait Some {}
impl Some for () {}
pub trait Other {}
impl Other for () {}

pub fn alef<T: Some>() -> T {
    loop {}
}
pub fn alpha() -> impl Some {}

pub fn bet<T, U>(t: T) -> U {
    loop {}
}
pub fn beta<T>(t: T) -> T {}

pub fn other<T: Other, U: Other>(t: T, u: U) {
    loop {}
}
pub fn alternate<T: Other>(t: T, u: T) {
    loop {}
}
