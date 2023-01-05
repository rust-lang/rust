macro_rules! m {
    ($s:stmt) => {}
}

m! { mut x }
//~^ ERROR expected expression, found keyword `mut`
//~| ERROR expected a statement
m! { auto x }
//~^ ERROR invalid variable declaration
m! { var x }
//~^ ERROR invalid variable declaration

fn main() {}
