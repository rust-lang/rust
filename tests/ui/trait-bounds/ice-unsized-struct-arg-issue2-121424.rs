// Regression test for issue #121424
#[repr(C)]
struct MySlice<T: Copy>(bool, T);
type MySliceBool = MySlice<[bool]>;
const MYSLICE_GOOD: &MySliceBool = &MySlice(true, [false]);
//~^ ERROR the trait bound `[bool]: Copy` is not satisfied
//~| ERROR the trait bound `[bool]: Copy` is not satisfied

fn main() {}
