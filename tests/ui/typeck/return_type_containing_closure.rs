#[allow(unused)]
fn foo() { //~ HELP try adding a return type
    vec!['a'].iter().map(|c| c)
    //~^ ERROR mismatched types [E0308]
    //~| NOTE expected `()`, found `Map<Iter<'_, char>, {closure@...}>`
    //~| NOTE expected unit type `()`
    //~| HELP consider using a semicolon here
}

fn main() {}
