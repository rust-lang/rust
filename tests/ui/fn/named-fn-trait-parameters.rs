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
    //~^ ERROR patterns aren't allowed in parenthesized argument list
    pat2: impl Fn((x, y): (bool, bool)),
    //~^ ERROR patterns aren't allowed in parenthesized argument list
    pat3: impl Fn(Thing { a, b }: Thing),
    //~^ ERROR patterns aren't allowed in parenthesized argument list
    pat4: impl Fn(NoThing { a, b }: NoThing),
    //~^ ERROR patterns aren't allowed in parenthesized argument list
    //~| ERROR cannot find type `NoThing` in this scope
    pat5: impl Fn((((((x))))): bool),
    //~^ ERROR patterns aren't allowed in parenthesized argument list

    self1: impl Fn(self),
    //~^ ERROR `self` parameter is only allowed in associated functions
    self2: impl Fn(self, self),
    //~^ ERROR `self` parameter is only allowed in associated functions
    //~| ERROR unexpected `self` parameter in function
    self3: impl Fn(bool, self),
    //~^ ERROR unexpected `self` parameter in function

    restricted_pat1: impl Fn(mut x: ()),
    //~^ ERROR patterns aren't allowed in parenthesized argument lists
    restricted_pat2: impl Fn(&x: ()),
    //~^ ERROR patterns aren't allowed in parenthesized argument lists
    restricted_pat3: impl Fn(&&x: ()),
    //~^ ERROR patterns aren't allowed in parenthesized argument lists
    restricted_pat4: impl Fn(false: ()),
    //~^ ERROR patterns aren't allowed in parenthesized argument lists
    restricted_pat5: impl Fn(&_: ()),
    //~^ ERROR patterns aren't allowed in parenthesized argument lists
    restricted_pat6: impl Fn(&true: ()),
    //~^ ERROR patterns aren't allowed in parenthesized argument lists
) { }

// Patterns are also syntactically rejected, but restricted patterns are not
#[cfg(false)]
fn syntax<F>(
    pat1: impl Fn(1..3: bool),
    //~^ ERROR patterns aren't allowed in parenthesized argument list
    pat2: impl Fn((x, y): (bool, bool)),
    //~^ ERROR patterns aren't allowed in parenthesized argument list
    pat3: impl Fn(Thing { a, b }: Thing),
    //~^ ERROR patterns aren't allowed in parenthesized argument list
    pat4: impl Fn(NoThing { a, b }: NoThing),
    //~^ ERROR patterns aren't allowed in parenthesized argument list
    pat5: impl Fn((((((x))))): bool),
    //~^ ERROR patterns aren't allowed in parenthesized argument list

    self1: impl Fn(self),
    self2: impl Fn(self, self),
    //~^ ERROR unexpected `self` parameter in function
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
