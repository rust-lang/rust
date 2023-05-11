struct Foo;
struct Bar;

impl Foo for Bar {} //~ ERROR E0404

fn main() {}

fn baz<T: Foo>(_: T) {} //~ ERROR E0404
