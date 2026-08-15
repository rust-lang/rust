//@ dont-require-annotations: NOTE

mod foo { pub mod foo {  } }

use foo::foo;
//~^ ERROR the name `foo` is defined multiple times
//~| NOTE `foo` reimported here

fn main() {}
