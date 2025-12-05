// Test for #95898: The trait suggestion had an extra `:` after the trait.
//@ edition:2021

fn foo<T:>(t: T) {
    t.clone();
    //~^ ERROR no method named `clone` found for type parameter `T` in the current scope
}

fn main() {}
