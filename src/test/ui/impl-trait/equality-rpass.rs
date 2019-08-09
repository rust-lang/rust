// run-pass

#![feature(specialization)]

trait Foo: std::fmt::Debug + Eq {}

impl<T: std::fmt::Debug + Eq> Foo for T {}

fn hide<T: Foo>(x: T) -> impl Foo {
    x
}

trait Leak<T>: Sized {
    fn leak(self) -> T;
}
impl<T, U> Leak<T> for U {
    default fn leak(self) -> T { panic!("type mismatch") }
}
impl<T> Leak<T> for T {
    fn leak(self) -> T { self }
}

trait CheckIfSend: Sized {
    type T: Default;
    fn check(self) -> Self::T { Default::default() }
}
impl<T> CheckIfSend for T {
    default type T = ();
}
impl<T: Send> CheckIfSend for T {
    type T = bool;
}

fn lucky_seven() -> impl Fn(usize) -> u8 {
    let a = [1, 2, 3, 4, 5, 6, 7];
    move |i| a[i]
}

fn main() {
    assert_eq!(hide(42), hide(42));

    assert_eq!(std::mem::size_of_val(&hide([0_u8; 5])), 5);
    assert_eq!(std::mem::size_of_val(&lucky_seven()), 7);

    assert_eq!(Leak::<i32>::leak(hide(5_i32)), 5_i32);

    assert_eq!(CheckIfSend::check(hide(0_i32)), false);
}
