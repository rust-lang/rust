trait Foo<const KIND: bool = true> {}
//~^ ERROR default values for const generic parameters are unstable

fn main() {}
