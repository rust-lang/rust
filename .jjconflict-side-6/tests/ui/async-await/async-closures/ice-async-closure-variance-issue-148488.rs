//@ edition: 2024

struct T<'g>();
//~^ ERROR lifetime parameter `'g` is never used

fn ord<a>() -> _ {
    //~^ WARN type parameter `a` should have an upper camel case name
    //~| ERROR the placeholder `_` is not allowed within types on item signatures for return types
    async || {}
}

fn main() {}
