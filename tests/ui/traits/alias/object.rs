// run-pass

#![feature(trait_alias)]

trait Foo = PartialEq<i32> + Send;
trait Bar = Foo + Sync;

trait I32Iterator = Iterator<Item = i32>;

pub fn main() {
    let a: &dyn Bar = &123;
    assert!(*a == 123);
    let b = Box::new(456) as Box<dyn Foo>;
    assert!(*b == 456);

    let c: &mut dyn I32Iterator = &mut vec![123].into_iter();
    assert_eq!(c.next(), Some(123));
}
