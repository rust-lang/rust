// Output of proposed syntax for anonymous enums from https://github.com/rust-lang/rfcs/issues/294.
// https://github.com/rust-lang/rust/issues/100741

fn foo(x: bool | i32) -> i32 | f64 {
    //~^ ERROR: function parameters require top-level or-patterns in parentheses
    //~| ERROR: expected one of `:`, `@`, or `|`, found `)`
    //~| ERROR: expected one of `!`, `(`, `)`, `+`, `,`, `::`, or `<`, found `|`
    //~| ERROR: expected one of `!`, `(`, `+`, `::`, `<`, `where`, or `{`, found `|`
    match x {
        x: i32 => x,
        true => 42.,
        false => 0.333,
    }
}

fn main() {
    match foo(true) {
        42: i32 => (),
        _: f64 => (),
        x: i32 => (),
    }
}
