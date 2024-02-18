union Foo {
    bar: i8,
    zst: (),
    pizza: Pizza,
}

#[derive(Clone, Copy)]
struct Pizza {
    topping: Option<PizzaTopping>
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
enum PizzaTopping {
    Cheese,
    Pineapple,
}

fn do_nothing(_x: &mut Foo) {}

pub fn main() {
    let mut foo = Foo { bar: 5 };
    do_nothing(&mut foo);

    // This is UB, so this test isn't run
    match foo {
        Foo { bar: _a } => {}, //~ ERROR access to union field is unsafe
    }
    match foo {
        Foo {
            pizza: Pizza { //~ ERROR access to union field is unsafe
                topping: Some(PizzaTopping::Cheese) | Some(PizzaTopping::Pineapple) | None
            }
        } => {},
    }

    match foo {
        Foo { zst: () } => {} //~ ERROR access to union field is unsafe
    }
    match foo {
        Foo { pizza: Pizza { .. } } => {} //~ ERROR access to union field is unsafe
    }

    // binding to wildcard is okay
    match foo {
        Foo { bar: _ } => {},
    }
    let Foo { bar: _ } = foo;
}
