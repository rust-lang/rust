// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #26656: Verify that trait objects cannot bypass dropck.

// Using this instead of Fn etc. to take HRTB out of the equation.
trait Trigger<B> { fn fire(&self, b: &mut B); }
impl<B: Button> Trigger<B> for () {
    fn fire(&self, b: &mut B) {
        b.push();
    }
}

// Still unsound Zook
trait Button { fn push(&self); }
struct Zook<B> { button: B, trigger: Box<Trigger<B>+'static> }

impl<B> Drop for Zook<B> {
    fn drop(&mut self) {
        self.trigger.fire(&mut self.button);
    }
}

// AND
struct Bomb { usable: bool }
impl Drop for Bomb { fn drop(&mut self) { self.usable = false; } }
impl Bomb { fn activate(&self) { assert!(self.usable) } }

enum B<'a> { HarmlessButton, BigRedButton(&'a Bomb) }
impl<'a> Button for B<'a> {
    fn push(&self) {
        if let B::BigRedButton(borrowed) = *self {
            borrowed.activate();
        }
    }
}

fn main() {
    let (mut zook, ticking);
    zook = Zook { button: B::HarmlessButton,
                  trigger: Box::new(()) };
    ticking = Bomb { usable: true };
    zook.button = B::BigRedButton(&ticking);
    //~^ ERROR `ticking` does not live long enough
}
