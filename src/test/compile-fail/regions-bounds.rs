// Check that explicit region bounds are allowed on the various
// nominal types (but not on other types) and that they are type
// checked.

enum an_enum = &int;
trait a_trait { fn foo() -> &self/int; }
struct a_class { let x:&self/int; new(x:&self/int) { self.x = x; } }

fn a_fn1(e: an_enum/&a) -> an_enum/&b {
    return e; //~ ERROR mismatched types: expected `an_enum/&b` but found `an_enum/&a`
}

fn a_fn2(e: a_trait/&a) -> a_trait/&b {
    return e; //~ ERROR mismatched types: expected `@a_trait/&b` but found `@a_trait/&a`
}

fn a_fn3(e: a_class/&a) -> a_class/&b {
    return e; //~ ERROR mismatched types: expected `a_class/&b` but found `a_class/&a`
}

fn a_fn4(e: int/&a) -> int/&b {
    //~^ ERROR region parameters are not allowed on this type
    //~^^ ERROR region parameters are not allowed on this type
    return e;
}

fn main() { }
