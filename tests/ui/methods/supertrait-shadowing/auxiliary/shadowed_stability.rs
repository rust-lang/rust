#![feature(staged_api)]
#![stable(feature = "main", since = "1.0.0")]

#[stable(feature = "main", since = "1.0.0")]
pub trait A {
    #[stable(feature = "main", since = "1.0.0")]
    fn hello(&self) {
        println!("A");
    }
}
#[stable(feature = "main", since = "1.0.0")]
impl<T> A for T {}

#[stable(feature = "main", since = "1.0.0")]
pub trait B: A {
    #[unstable(feature = "downstream", issue = "none")]
    fn hello(&self) {
        println!("B");
    }
}
#[stable(feature = "main", since = "1.0.0")]
impl<T> B for T {}
