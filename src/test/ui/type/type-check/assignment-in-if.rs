// Test that the parser does not attempt to parse struct literals
// within assignments in if expressions.

#![allow(unused_parens)]

struct Foo {
    foo: usize
}

fn main() {
    let x = 1;
    let y: Foo;

    // `x { ... }` should not be interpreted as a struct literal here
    if x = x {
        //~^ ERROR mismatched types
        println!("{}", x);
    }
    // Explicit parentheses on the left should match behavior of above
    if (x = x) {
        //~^ ERROR mismatched types
        println!("{}", x);
    }
    // The struct literal interpretation is fine with explicit parentheses on the right
    if y = (Foo { foo: x }) {
        //~^ ERROR mismatched types
        println!("{}", x);
    }
    // "invalid left-hand side expression" error is suppresed
    if 3 = x {
        //~^ ERROR mismatched types
        println!("{}", x);
    }
    if (
        if true {
            x = 4 //~ ERROR mismatched types
        } else {
            x = 5 //~ ERROR mismatched types
        }
    ) {
        println!("{}", x);
    }
}
