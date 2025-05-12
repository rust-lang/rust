// Shadowing a unit-like enum in a closure

struct Test;

fn main() {
    || {
        let Test = 1; //~ ERROR mismatched types
    };
}
