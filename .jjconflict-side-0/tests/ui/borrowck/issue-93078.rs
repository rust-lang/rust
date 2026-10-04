trait Modify {
    fn modify(&mut self) ;
}

impl<T> Modify for T  {
    fn modify(&mut self)  {}
}

trait Foo {
    fn mute(&mut self) {
        self.modify(); //~ ERROR cannot borrow `self` as mutable
    }
}

fn main() {}
