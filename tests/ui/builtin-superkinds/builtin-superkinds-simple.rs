// Basic test for traits inheriting from the builtin kinds, checking
// the type contents of the implementing type (that's not a typaram).

trait Foo : Send { }

impl Foo for std::rc::Rc<i8> { }
//~^ ERROR `Rc<i8>` cannot be sent between threads safely

fn main() { }
