// rustfmt-wrap_match_arms: false
// Match expressions, no unwrapping of block arms or wrapping of multiline
// expressions.

fn foo() {
    match x {
        a => {
            foo()
        }
        b => (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
              bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb),
    }
}
