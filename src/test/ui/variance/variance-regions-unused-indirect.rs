// Test that disallow lifetime parameters that are unused.

enum Foo<'a> { //~ ERROR parameter `'a` is never used
    Foo1(Bar<'a>)
}

enum Bar<'a> { //~ ERROR parameter `'a` is never used
    Bar1(Foo<'a>)
}

fn main() {}
