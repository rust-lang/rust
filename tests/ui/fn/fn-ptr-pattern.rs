fn patterns<F>(
    pat1: fn(true: bool),
    //~^ ERROR patterns aren't allowed in function pointer types
    pat2: fn(1..3: bool),
    //~^ ERROR patterns aren't allowed in function pointer types
    pat3: fn((x, y): (bool, bool)),
    //~^ ERROR patterns aren't allowed in function pointer types
    pat4: fn(self),
    //~^ ERROR `self` parameter is only allowed in associated functions
    pat5: fn(self, self),
    //~^ ERROR `self` parameter is only allowed in associated functions
    //~| ERROR unexpected `self` parameter in function
    pat6: fn(bool, self),
    //~^ ERROR unexpected `self` parameter in function
    pat7: fn(Thing { a, b }: Thing),
    //~^ ERROR patterns aren't allowed in function pointer types
    pat8: fn(NoThing { a, b }: NoThing),
    //~^ ERROR patterns aren't allowed in function pointer types
    //~| ERROR cannot find type `NoThing` in this scope
    pat9: fn((((((x))))): bool),
    //~^ ERROR patterns aren't allowed in function pointer types
) { }

struct Thing { a: bool, b: bool }

fn main() {

}
