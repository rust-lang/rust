//@ compile-flags: --crate-type=lib

trait X { fn dummy(&self) { } }
impl X for usize { }

trait Y { fn dummy(&self) { } }
impl Y for usize { }
