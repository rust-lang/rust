#![feature(staged_api)]
#![stable(feature = "test", since = "1.0.0")]

#[stable(feature = "test", since = "1.0.0")]
pub trait Unstable<T> {
    #[unstable(feature = "unstable_value", issue = "none")]
    fn value(&self) -> T;
}

#[stable(feature = "test", since = "1.0.0")]
impl Unstable<u8> for () {
    fn value(&self) -> u8 {
        8
    }
}

#[stable(feature = "test", since = "1.0.0")]
impl Unstable<u16> for () {
    fn value(&self) -> u16 {
        16
    }
}
