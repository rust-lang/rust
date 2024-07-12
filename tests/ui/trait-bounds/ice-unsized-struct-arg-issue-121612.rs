// Regression test for issue #121612

trait Trait {}
impl Trait for bool {}
struct MySlice<T: FnOnce(&T, Idx) -> Idx>(bool, T);
//~^ ERROR cannot find type `Idx`
//~| ERROR cannot find type `Idx`
type MySliceBool = MySlice<[bool]>;
const MYSLICE_GOOD: &MySliceBool = &MySlice(true, [false]);
//~^ ERROR the size for values of type `[bool]` cannot be known at compilation time
//~| ERROR the size for values of type `[bool]` cannot be known at compilation time

fn main() {}
