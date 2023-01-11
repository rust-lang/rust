// run-pass

#![feature(trait_alias)]

use std::marker::PhantomData;

trait Empty {}
trait EmptyAlias = Empty;
trait CloneDefault = Clone + Default;
trait SendSyncAlias = Send + Sync;
trait WhereSendAlias = where Self: Send;
trait SendEqAlias<T> = Send where T: PartialEq<Self>;
trait I32Iterator = Iterator<Item = i32>;

#[allow(dead_code)]
struct Foo<T: SendSyncAlias>(PhantomData<T>);
#[allow(dead_code)]
struct Bar<T>(PhantomData<T>) where T: SendSyncAlias;

impl dyn EmptyAlias {}

impl<T: SendSyncAlias> Empty for T {}

fn a<T: CloneDefault>() -> (T, T) {
    let one = T::default();
    let two = one.clone();
    (one, two)
}

fn b(x: &impl SendEqAlias<i32>) -> bool {
    22_i32 == *x
}

fn c<T: I32Iterator>(x: &mut T) -> Option<i32> {
    x.next()
}

fn d<T: SendSyncAlias>() {
    is_send_and_sync::<T>();
}

fn is_send_and_sync<T: Send + Sync>() {}

fn main() {
    let both = a::<i32>();
    assert_eq!(both.0, 0);
    assert_eq!(both.1, 0);
    let both: (i32, i32) = a();
    assert_eq!(both.0, 0);
    assert_eq!(both.1, 0);

    assert!(b(&22));

    assert_eq!(c(&mut vec![22].into_iter()), Some(22));

    d::<i32>();
}
