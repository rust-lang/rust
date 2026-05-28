// Pin's PartialEq implementation allowed to access the pointer allowing for
// unsoundness by using Rc::get_mut to move value within Rc.
// See https://internals.rust-lang.org/t/unsoundness-in-pin/11311/73 for more details.

use std::ops::Deref;
use std::pin::Pin;
use std::rc::Rc;

struct Apple;

impl Deref for Apple {
    type Target = Apple;
    fn deref(&self) -> &Apple {
        &Apple
    }
}

impl PartialEq<Rc<Apple>> for Apple {
    fn eq(&self, _rc: &Rc<Apple>) -> bool {
        unreachable!()
    }
}

fn main() {
    let _ = Pin::new(Apple) == Rc::pin(Apple);
    //~^ ERROR type mismatch resolving
}
