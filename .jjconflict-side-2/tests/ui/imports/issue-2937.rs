use m::f as x; //~ ERROR unresolved import `m::f` [E0432]
               //~^ NOTE no `f` in `m`

mod m {}

fn main() {}
