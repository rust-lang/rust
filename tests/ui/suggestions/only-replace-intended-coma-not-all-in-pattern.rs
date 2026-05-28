//@ run-rustfix

// Regression test for issue #143330.
// Ensure we suggest to replace only the intended coma with a bar, not all commas in the pattern.

fn main() {
    struct Foo { x: i32, ch: char }
    let pos = Foo { x: 2, ch: 'x' };
    match pos {
        // All commas here were replaced with bars.
        // Foo { x: 2 | ch: ' |' } | Foo { x: 3 | ch: '@' } => (),
        Foo { x: 2, ch: ',' }, Foo { x: 3, ch: '@' } => (),
        //~^ ERROR unexpected `,` in pattern
        //~| HELP try adding parentheses to match on a tuple...
        //~| HELP ...or a vertical bar to match on alternative
        _ => todo!(),
    }
}
