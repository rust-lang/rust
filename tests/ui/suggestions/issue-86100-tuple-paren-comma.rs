// Tests that a suggestion is issued for type mismatch errors when a
// 1-tuple is expected and a parenthesized expression of non-tuple
// type is supplied.

fn foo<T>(_t: (T,)) {}
struct S { _s: (String,) }

fn main() {
    let _x: (i32,) = (5);
    //~^ ERROR: mismatched types [E0308]
    //~| HELP: use a trailing comma to create a tuple with one element

    foo((Some(3)));
    //~^ ERROR: mismatched types [E0308]
    //~| HELP: use a trailing comma to create a tuple with one element

    let _s = S { _s: ("abc".to_string()) };
    //~^ ERROR: mismatched types [E0308]
    //~| HELP: use a trailing comma to create a tuple with one element

    // Do not issue the suggestion if the found type is already a tuple.
    let t = (1, 2);
    let _x: (i32,) = (t);
    //~^ ERROR: mismatched types [E0308]
}
