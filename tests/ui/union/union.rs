union Foo {
    bar: i8,
    zst: (),
    tuple: (i32,),
    pizza: Pizza,
    tuple_struct: TupleStruct,
}

#[derive(Clone, Copy)]
struct Pizza {
    topping: Option<PizzaTopping>,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
enum PizzaTopping {
    Cheese,
    Pineapple,
}

#[derive(Clone, Copy)]
struct TupleStruct(i32);

fn do_nothing(_x: &mut Foo) {}

pub fn main() {
    let mut foo = Foo { bar: 5 };
    do_nothing(&mut foo);

    // This is UB, so this test isn't run
    match foo {
        Foo { bar: _a } => {} //~ ERROR access to union field is unsafe
    }
    match foo {
        Foo {
            pizza:
                Pizza {
                    topping: Some(PizzaTopping::Cheese) | Some(PizzaTopping::Pineapple) | None,
                    //~^ ERROR access to union field is unsafe
                    //~| ERROR access to union field is unsafe
                    //~| ERROR access to union field is unsafe
                },
        } => {}
    }
    match foo {
        Foo { tuple: (_a,) } => {} //~ ERROR access to union field is unsafe
    }
    match foo {
        Foo { tuple_struct: TupleStruct(_a) } => {} //~ ERROR access to union field is unsafe
    }

    // binding to a struct pattern is okay if no fields are read
    match foo {
        Foo { zst: () } => {}
    }
    match foo {
        Foo { zst: (..) } => {}
    }
    match foo {
        Foo { tuple: (..) } => {}
    }
    match foo {
        Foo { tuple: (_,) } => {}
    }
    match foo {
        Foo { pizza: Pizza { .. } } => {}
    }
    match foo {
        Foo { pizza: Pizza { topping: _ } } => {}
    }
    match foo {
        Foo { tuple_struct: TupleStruct(_) } => {}
    }

    // binding to wildcard is okay
    match foo {
        Foo { bar: _ } => {}
    }
    let Foo { bar: _ } = foo;
}
