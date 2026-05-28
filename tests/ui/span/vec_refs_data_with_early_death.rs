// This test is a simple example of code that violates the dropck
// rules: it pushes `&x` and `&y` into a bag (with dtor), but the
// referenced data will be dropped before the bag is.







fn main() {
    let mut v = Bag::new();

    let x: i8 = 3;
    let y: i8 = 4;

    v.push(&x);
    //~^ ERROR `x` does not live long enough
    v.push(&y);
    //~^ ERROR `y` does not live long enough

    assert_eq!(v.0, [&3, &4]);
}

//`Vec<T>` is #[may_dangle] w.r.t. `T`; putting a bag over its head
// forces borrowck to treat dropping the bag as a potential use.
struct Bag<T>(Vec<T>);
impl<T> Drop for Bag<T> { fn drop(&mut self) { } }

impl<T> Bag<T> {
    fn new() -> Self { Bag(Vec::new()) }
    fn push(&mut self, t: T) { self.0.push(t); }
}
