fn foo<T: Into<String>>(x: i32) {}
//~^ NOTE required by

fn main() {
    foo(42);
    //~^ ERROR type annotations needed
    //~| NOTE cannot infer type for type parameter `T` declared on the function `foo`
    //~| NOTE cannot satisfy
}
