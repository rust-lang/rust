trait WhereTrait {
    type Type;
}

fn foo(e: Enum) {
    if let Enum::Map(_) = e {}

    match e {
        //~^ ERROR: non-exhaustive patterns: `Enum::Map2(_)` not covered
        Enum::Map(_) => (),
    }
}

enum Enum {
    Map(()),
    Map2(<() as WhereTrait>::Type),
    //~^ ERROR: the trait bound `(): WhereTrait` is not satisfied
}

fn main() {}
