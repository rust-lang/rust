// Test that disallow lifetime parameters that are unused.

enum Foo<'a> { //~ ERROR parameter `'a` is never used
    //~^ ERROR recursive types `Foo` and `Bar` have infinite size
    Foo1(Bar<'a>)
}

enum Bar<'a> { //~ ERROR parameter `'a` is never used
    Bar1(Foo<'a>)
}

fn main() {}
