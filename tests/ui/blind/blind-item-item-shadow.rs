mod foo { pub mod foo {  } }

use foo::foo;
//~^ ERROR the name `foo` is defined multiple times
//~| NOTE_NONVIRAL `foo` reimported here

fn main() {}
