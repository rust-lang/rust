trait X {
    type Y<'a>;
}

fn f(x: Box<dyn X<Y<'a> = &'a ()>>) {}
//~^ ERROR: use of undeclared lifetime name `'a`
//~| ERROR: use of undeclared lifetime name `'a`
//~| ERROR: the trait `X` cannot be made into an object [E0038]

fn main() {}
