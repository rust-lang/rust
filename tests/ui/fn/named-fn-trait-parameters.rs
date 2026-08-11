#![feature(named_fn_trait_parameters)]

fn allowed<F>(
    data: &str,
    f1: impl Fn(msg: String),
    f2: impl Fn(_: String),
    f3: impl Fn(String, msg: String),
    f4: impl Fn(msg: String, String),
    f5: impl Fn(duplicate_name: bool, duplicate_name: bool),
    fg: F
) where F: Fn(msg: String)
{ }


// Patterns are semantically rejected
fn semantics<F>(
    pat1: impl Fn(1..3: bool),
    //~^ ERROR expected type, found `1`
    pat2: impl Fn((x, y): (bool, bool)),
    //~^ ERROR unexpected token: `:`
    pat3: impl Fn(Thing { a, b }: Thing),
    //~^ ERROR expected one of `!`, `(`, `+`, `::`, or `<`, found `{`
    pat4: impl Fn(NoThing { a, b }: NoThing),
    //~^ ERROR expected one of `!`, `(`, `+`, `::`, or `<`, found `{`
    pat5: impl Fn((((((x))))): bool),
    //~^ ERROR unexpected token: `:`

    self1: impl Fn(self),
    //~^ ERROR unexpected `self` parameter in function
    self2: impl Fn(self, self),
    //~^ ERROR unexpected `self` parameter in function
    //~| ERROR unexpected `self` parameter in function
    self3: impl Fn(bool, self),
    //~^ ERROR unexpected `self` parameter in function

    // FIXME should be rejected
    restricted_pat1: impl Fn(mut x: ()),
    restricted_pat2: impl Fn(&x: ()),
    restricted_pat3: impl Fn(&&x: ()),
    restricted_pat4: impl Fn(false: ()),
    restricted_pat5: impl Fn(&_: ()),
    restricted_pat6: impl Fn(&true: ()),
) { }

// Patterns are also syntactically rejected, but restricted patterns are not
#[cfg(false)]
fn syntax<F>(
    pat1: impl Fn(1..3: bool),
    //~^ ERROR expected type, found `1`
    pat2: impl Fn((x, y): (bool, bool)),
    //~^ ERROR unexpected token: `:`
    pat3: impl Fn(Thing { a, b }: Thing),
    //~^ ERROR expected one of `!`, `(`, `+`, `::`, or `<`, found `{`
    pat4: impl Fn(NoThing { a, b }: NoThing),
    //~^ ERROR expected one of `!`, `(`, `+`, `::`, or `<`, found `{`
    pat5: impl Fn((((((x))))): bool),
    //~^ ERROR unexpected token: `:`

    self1: impl Fn(self), // FIXME should be accepted
    //~^ ERROR unexpected `self` parameter in function
    self2: impl Fn(self, self),
    //~^ ERROR unexpected `self` parameter in function
    //~| ERROR unexpected `self` parameter in function
    self3: impl Fn(bool, self),
    //~^ ERROR unexpected `self` parameter in function

    // These are correctly accepted, as to match the behaviour `fn` ptrs
    restricted_pat1: impl Fn(mut x: ()),
    restricted_pat2: impl Fn(&x: ()),
    restricted_pat3: impl Fn(&&x: ()),
    restricted_pat4: impl Fn(false: ()),
    restricted_pat5: impl Fn(&_: ()),
    restricted_pat6: impl Fn(&true: ()),
) { }

struct Thing { a: bool, b: bool }

fn main() {

}
