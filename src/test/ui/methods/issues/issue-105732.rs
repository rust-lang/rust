#![feature(auto_traits)]

auto trait Foo {
    fn g(&self); //~ ERROR auto traits cannot have associated items
}

trait Bar {
    fn f(&self) {
        self.g(); //~ ERROR the method `g` exists for reference `&Self`, but its trait bounds were not satisfied
    }
}

fn main() {}
