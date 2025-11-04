type T = dyn core::fmt::Debug;
 
fn f() -> T { loop {} }
//~^ ERROR return type cannot be a trait object without pointer indirection
//~| HELP 
//~| HELP 

fn main() {}
