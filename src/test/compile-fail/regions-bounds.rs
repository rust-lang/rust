// Check that explicit region bounds are allowed on the various
// nominal types (but not on other types) and that they are type
// checked.

enum an_enum = &int;
iface an_iface { fn foo() -> &self/int; }
class a_class { let x:&self/int; new(x:&self/int) { self.x = x; } }

fn a_fn1(e: an_enum/&a) -> an_enum/&b {
    ret e; //~ ERROR mismatched types: expected `an_enum/&b` but found `an_enum/&a`
}

fn a_fn2(e: an_iface/&a) -> an_iface/&b {
    ret e; //~ ERROR mismatched types: expected `an_iface/&b` but found `an_iface/&a`
}

fn a_fn3(e: a_class/&a) -> a_class/&b {
    ret e; //~ ERROR mismatched types: expected `a_class/&b` but found `a_class/&a`
}

fn a_fn4(e: int/&a) -> int/&b {
    //~^ ERROR region parameters are not allowed on this type
    //~^^ ERROR region parameters are not allowed on this type
    ret e;
}

fn main() { }