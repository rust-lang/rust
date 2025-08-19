//@revisions: edition2015 edition2024
//@[edition2015] edition:2015
//@[edition2024] edition:2024
#![feature(impl_trait_in_assoc_type)]

trait Callable {
    type Output;
    fn call(x: Self) -> Self::Output;
}

trait PlusOne {
    fn plus_one(&mut self);
}

impl<'a> PlusOne for &'a mut i32 {
    fn plus_one(&mut self) {
        **self += 1;
    }
}

impl<T: PlusOne> Callable for T {
    type Output = impl PlusOne;
    fn call(t: T) -> Self::Output {
        t
    }
}

fn test<'a>(y: &'a mut i32) -> impl PlusOne {
    <&'a mut i32 as Callable>::call(y)
    //[edition2015]~^ ERROR hidden type for `impl PlusOne` captures lifetime that does not appear in bounds
}

fn main() {
    let mut z = 42;
    let mut thing = test(&mut z);
    let mut thing2 = test(&mut z); //[edition2024]~ ERROR cannot borrow `z` as mutable more than once
    thing.plus_one();
    assert_eq!(z, 43); //[edition2024]~ ERROR cannot borrow `z` as immutable
    thing2.plus_one();
    assert_eq!(z, 44); //[edition2024]~ ERROR cannot borrow `z` as immutable
    thing.plus_one();
    assert_eq!(z, 45); //[edition2024]~ ERROR cannot borrow `z` as immutable
}
