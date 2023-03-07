trait Foo {
    fn foo();
}

use Foo::foo; //~ ERROR not directly importable

fn main() { foo(); }
