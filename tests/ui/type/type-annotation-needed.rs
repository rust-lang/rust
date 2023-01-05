fn foo<T: Into<String>>(x: i32) {}
//~^ NOTE required by
//~| NOTE required by

fn main() {
    foo(42);
    //~^ ERROR type annotations needed
    //~| NOTE cannot infer type
    //~| NOTE cannot satisfy
}
