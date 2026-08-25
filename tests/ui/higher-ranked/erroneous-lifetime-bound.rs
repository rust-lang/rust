fn foo<T>() where T: for<'a> 'a {}
//~^ ERROR `for<...>` may only modify trait bounds, not lifetime bounds
//~| ERROR use of undeclared lifetime name `'a` [E0261]

fn main() {}
