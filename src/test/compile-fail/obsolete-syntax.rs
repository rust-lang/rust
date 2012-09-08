fn f1<T: copy>() -> T { }
//~^ ERROR obsolete syntax: lower-case kind bounds

fn f1<T: send>() -> T { }
//~^ ERROR obsolete syntax: lower-case kind bounds

fn f1<T: const>() -> T { }
//~^ ERROR obsolete syntax: lower-case kind bounds

fn f1<T: owned>() -> T { }
//~^ ERROR obsolete syntax: lower-case kind bounds

struct s {
    let foo: (),
    //~^ ERROR obsolete syntax: `let` in field declaration
    bar: ();
    //~^ ERROR obsolete syntax: field declaration terminated with semicolon
    new() { }
    //~^ ERROR obsolete syntax: struct constructor
}

fn obsolete_with() {
    struct S {
        foo: (),
        bar: (),
    }

    let a = S { foo: (), bar: () };
    let b = S { foo: () with a };
    //~^ ERROR obsolete syntax: with
    let c = S { foo: (), with a };
    //~^ ERROR obsolete syntax: with
    let a = { foo: (), bar: () };
    let b = { foo: () with a };
    //~^ ERROR obsolete syntax: with
    let c = { foo: (), with a };
    //~^ ERROR obsolete syntax: with
}

fn main() { }
