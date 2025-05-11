trait Foo {
    fn foo();
}

use Foo::foo; //~ ERROR `use` associated items of traits is unstable [E0658]

fn main() { foo(); } //~ ERROR type annotations needed
