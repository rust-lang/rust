fn foo<T>() where T: Default -> impl Default + 'static {}
//~^ ERROR return type should be specified after the function parameters
//~| HELP place the return type after the function parameters

fn main() {}
