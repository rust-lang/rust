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

    reuse S::<(), 123>::by_ref { self.get_s() }
    //~^ ERROR: cannot move out of `*self` which is behind a shared reference

    reuse S::<(), 123>::by_mut_ref { self.get_s() }
    //~^ ERROR: cannot move out of `*self` which is behind a mutable reference

    reuse S::<(), 123>::by_box { self.get_s() }
    //~^ ERROR: mismatched types

    reuse S::<(), 123>::by_rc { self.get_s() }
    //~^ ERROR: mismatched types

    reuse S::<(), 123>::by_pin { self.get_s() }
    //~^ ERROR: mismatched types
}

trait Trait2: Sized {
    reuse S::<(), 123>::by_value { self.get_s() }
    //~^ ERROR: no method named `get_s` found for type parameter `Self` in the current scope
    reuse S::<(), 123>::by_ref { self.get_s() }
    //~^ ERROR: no method named `get_s` found for reference `&Self` in the current scope
    reuse S::<(), 123>::by_mut_ref { self.get_s() }
    //~^ ERROR: no method named `get_s` found for mutable reference `&mut Self` in the current scope
    reuse S::<(), 123>::by_box { self.get_s() }
    //~^ ERROR: no method named `get_s` found for struct `Box<Self>` in the current scope
    reuse S::<(), 123>::by_rc { self.get_s() }
    //~^ ERROR: no method named `get_s` found for struct `Rc<Self>` in the current scope
    reuse S::<(), 123>::by_pin { self.get_s() }
    //~^ ERROR: no method named `get_s` found for struct `Pin<Box<Self>>` in the current scope
}

fn main() {}
