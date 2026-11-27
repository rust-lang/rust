// Tests that we consider `Box<U>: !Sugar` to be ambiguous, even
// though we see no impl of `Sugar` for `Box`. Therefore, an overlap
// error is reported for the following pair of impls (#23516).

pub trait Sugar {}

struct Cake<X>(X);

impl<T:Sugar> Cake<T> { fn dummy(&self) { } }
//~^ ERROR E0592
impl<U:Sugar> Cake<Box<U>> { fn dummy(&self) { } }

fn main() { }
