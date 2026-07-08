struct Foo<'a, A> {}
//~^ ERROR parameter `'a` is never used
//~| ERROR parameter `A` is never used

fn main() {}
