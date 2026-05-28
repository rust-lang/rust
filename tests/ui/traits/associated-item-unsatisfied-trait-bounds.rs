struct Foo;
trait Bar {}
trait Baz {}
trait Bat { fn bat(&self); }
impl<T> Bat for T where T: 'static + Bar + Baz { fn bat(&self) { println!("generic bat"); } }

pub fn main() {
    Foo::bat(()); //~ ERROR E0599
}
