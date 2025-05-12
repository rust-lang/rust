use foo::self; //~ ERROR unresolved import `foo`
//~^ ERROR `self` imports are only allowed within a { } list

use std::mem::self;
//~^ ERROR `self` imports are only allowed within a { } list

fn main() {}
