struct Inv<'a>(&'a mut &'a ());
enum Foo<T> {
    Bar,
    Var(T),
}
type Supertype = Foo<for<'a> fn(Inv<'a>, Inv<'a>)>;

fn foo(x: Foo<for<'a, 'b> fn(Inv<'a>, Inv<'b>)>) {
    match x {
        Supertype::Bar => {}
        //~^ ERROR mismatched types
        //~| ERROR mismatched types
        Supertype::Var(x) => {}
        //~^ ERROR mismatched types
        //~| ERROR mismatched types
        //~| ERROR mismatched types
        //~| ERROR mismatched types
    }
}

fn foo_nested(x: Foo<Foo<for<'a, 'b> fn(Inv<'a>, Inv<'b>)>>) {
    match x {
    //~^ ERROR mismatched types
    //~| ERROR mismatched types
        Foo::Bar => {}
        Foo::Var(Supertype::Bar) => {}
        //~^ ERROR mismatched types
        //~| ERROR mismatched types
        Foo::Var(Supertype::Var(x)) => {}
        //~^ ERROR mismatched types
        //~| ERROR mismatched types
        //~| ERROR mismatched types
        //~| ERROR mismatched types
    }
}

fn main() {}
