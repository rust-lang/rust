// rustfmt-match_block_trailing_comma: true
// Match expressions, no unwrapping of block arms or wrapping of multiline
// expressions.

fn foo() {
    match x {
        a => {
            "line1";
            "line2"
        }
        b => (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
              bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb),
    }
}
