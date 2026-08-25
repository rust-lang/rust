#![feature(fn_delegation)]

use std::rc::Rc;
use std::pin::Pin;

struct S<A, const C: usize> {
    xd: [A; C],
}

impl<'a, 'b, 'c, A, const C: usize> S<A, C> {
    fn by_value<'d: 'd, 'e, T, const B: bool>(self) {}
    fn by_ref<'d: 'd, 'e, T, const B: bool>(&self) {}
    fn by_mut_ref<'d: 'd, 'e, T, const B: bool>(&mut self) {}
    fn by_box<'d: 'd, 'e, T, const B: bool>(self: Box<Self>) {}
    fn by_rc<'d: 'd, 'e, T, const B: bool>(self: Rc<Self>) {}
    fn by_pin<'d: 'd, 'e, T, const B: bool>(self: Pin<Box<Self>>) {}
}

trait Trait: Sized {
    fn get_s(self) -> S<(), 123>;

    reuse S::<(), 123>::by_value { self.get_s() }
    //~^ ERROR: cannot find function `by_value` in `S`

    reuse S::<(), 123>::by_ref { self.get_s() }
    //~^ ERROR: cannot find function `by_ref` in `S`

    reuse S::<(), 123>::by_mut_ref { self.get_s() }
    //~^ ERROR: cannot find function `by_mut_ref` in `S`

    reuse S::<(), 123>::by_box { self.get_s() }
    //~^ ERROR: cannot find function `by_box` in `S`

    reuse S::<(), 123>::by_rc { self.get_s() }
    //~^ ERROR: cannot find function `by_rc` in `S`

    reuse S::<(), 123>::by_pin { self.get_s() }
    //~^ ERROR: cannot find function `by_pin` in `S`
}

trait Trait2: Sized {
    reuse S::<(), 123>::by_value { self.get_s() }
    //~^ ERROR: cannot find function `by_value` in `S`
    reuse S::<(), 123>::by_ref { self.get_s() }
    //~^ ERROR: cannot find function `by_ref` in `S`
    reuse S::<(), 123>::by_mut_ref { self.get_s() }
    //~^ ERROR: cannot find function `by_mut_ref` in `S`
    reuse S::<(), 123>::by_box { self.get_s() }
    //~^ ERROR: cannot find function `by_box` in `S`
    reuse S::<(), 123>::by_rc { self.get_s() }
    //~^ ERROR: cannot find function `by_rc` in `S`
    reuse S::<(), 123>::by_pin { self.get_s() }
    //~^ ERROR: cannot find function `by_pin` in `S`
}

fn main() {}
