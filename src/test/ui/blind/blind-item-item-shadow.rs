mod foo { pub mod foo {  } }

use foo::foo;
//~^ ERROR the name `foo` is defined multiple times
//~| `foo` reimported here

fn main() {}
