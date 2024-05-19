//@ ignore-macos: cycle error does not appear on apple

use std::sync::Mutex;

enum Foo { X(Mutex<Option<Foo>>) }
//~^ ERROR recursive type `Foo` has infinite size
//~| ERROR cycle detected

impl Foo { fn bar(self) {} }

fn main() {}
