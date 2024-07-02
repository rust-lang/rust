// these two HELPs are actually in a new line between this line and the `enum Fruit` line
enum Fruit {
    Apple(i64),
    Orange(i64),
}

fn should_return_fruit() -> Apple {
    //~^ ERROR cannot find type `Apple`
    Apple(5)
    //~^ ERROR cannot find function, tuple struct or tuple variant `Apple`
}

fn should_return_fruit_too() -> Fruit::Apple {
    //~^ ERROR expected type, found variant `Fruit::Apple`
    Apple(5)
    //~^ ERROR cannot find function, tuple struct or tuple variant `Apple`
}

fn foo() -> Ok {
    //~^ ERROR expected type, found variant `Ok`
    Ok(())
}

fn bar() -> Variant3 {
    //~^ ERROR cannot find type `Variant3`
}

fn qux() -> Some {
    //~^ ERROR expected type, found variant `Some`
    Some(1)
}

fn main() {}

mod x {
    pub enum Enum {
        Variant1,
        Variant2(),
        Variant3(usize),
        Variant4 {},
    }
}
