#![feature(auto_traits)]

auto trait Foo {
    fn g(&self); //~ ERROR auto traits cannot have associated items
}

trait Bar {
    fn f(&self) {
        // issue #105788
        self.g(); //~ ERROR no method named `g` found for reference `&Self` in the current scope
    }
}

fn main() {}
