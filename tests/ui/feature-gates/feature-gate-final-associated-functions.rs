trait Foo {
    final fn bar() {}
    //~^ ERROR `final` on trait functions is experimental
}

fn main() {}
