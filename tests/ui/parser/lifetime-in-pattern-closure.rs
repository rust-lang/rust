// Test case for issue #134061
// Check that the compiler properly suggests removing unexpected lifetime in closure pattern

const x: () = |&'a| ();
//~^ ERROR unexpected lifetime `'a` in pattern
//~| HELP remove the lifetime
//~| ERROR expected parameter name, found `|`

fn main() {}
