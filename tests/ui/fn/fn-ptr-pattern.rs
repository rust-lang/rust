fn allowed<F>(
    data: &str,
    f1: fn(msg: String),
    f2: fn(_: String),
    f3: fn(String, msg: String),
    f4: fn(msg: String, String),
    f5: fn(duplicate_name: bool, duplicate_name: bool),
) { }


// Patterns are semantically rejected
fn semantics<F>(
    pat1: fn(1..3: bool),
    //~^ ERROR patterns aren't allowed in function pointer types
    pat2: fn((x, y): (bool, bool)),
    //~^ ERROR patterns aren't allowed in function pointer types
    pat3: fn(Thing { a, b }: Thing),
    //~^ ERROR patterns aren't allowed in function pointer types
    pat4: fn(NoThing { a, b }: NoThing),
    //~^ ERROR patterns aren't allowed in function pointer types
    //~| ERROR cannot find type `NoThing` in this scope
    pat5: fn((((((x))))): bool),
    //~^ ERROR patterns aren't allowed in function pointer types

    self1: fn(self),
    //~^ ERROR `self` parameter is only allowed in associated functions
    self2: fn(self, self),
    //~^ ERROR `self` parameter is only allowed in associated functions
    //~| ERROR unexpected `self` parameter in function
    self3: fn(bool, self),
    //~^ ERROR unexpected `self` parameter in function

    restricted_pat1: fn(mut x: ()),
    //~^ ERROR patterns aren't allowed in function pointer types
    restricted_pat2: fn(&x: ()),
    //~^ ERROR patterns aren't allowed in function pointer types
    restricted_pat3: fn(&&x: ()),
    //~^ ERROR patterns aren't allowed in function pointer types
    restricted_pat4: fn(false: ()),
    //~^ ERROR patterns aren't allowed in function pointer types
    restricted_pat5: fn(&_: ()),
    //~^ ERROR patterns aren't allowed in function pointer types
    restricted_pat6: fn(&true: ()),
    //~^ ERROR patterns aren't allowed in function pointer types
) { }

// Patterns are also syntactically rejected, but restricted patterns are not
#[cfg(false)]
fn syntax<F>(
    pat1: fn(1..3: bool),
    //~^ ERROR patterns aren't allowed in function pointer types
    pat2: fn((x, y): (bool, bool)),
    //~^ ERROR patterns aren't allowed in function pointer types
    pat3: fn(Thing { a, b }: Thing),
    //~^ ERROR patterns aren't allowed in function pointer types
    pat4: fn(NoThing { a, b }: NoThing),
    //~^ ERROR patterns aren't allowed in function pointer types
    pat5: fn((((((x))))): bool),
    //~^ ERROR patterns aren't allowed in function pointer types

    self1: fn(self),
    self2: fn(self, self),
    //~^ ERROR unexpected `self` parameter in function
    self3: fn(bool, self),
    //~^ ERROR unexpected `self` parameter in function

    restricted_pat1: fn(mut x: ()),
    restricted_pat2: fn(&x: ()),
    restricted_pat3: fn(&&x: ()),
    restricted_pat4: fn(false: ()),
    restricted_pat5: fn(&_: ()),
    restricted_pat6: fn(&true: ()),
) { }

struct Thing { a: bool, b: bool }

fn main() {

}
