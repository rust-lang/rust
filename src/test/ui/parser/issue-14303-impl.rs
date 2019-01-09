struct X<T>(T);

impl<'a, T, 'b> X<T> {}
//~^ ERROR lifetime parameters must be declared prior to type parameters

fn main() {}
