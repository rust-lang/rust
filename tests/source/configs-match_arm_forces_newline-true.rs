// rustfmt-match_arm_forces_newline: true
// rustfmt-wrap_match_arms: false

// match_arm_forces_newline puts all match arms bodies in a newline and indents
// them.

fn main() {
    match x() {
        // a short non-empty block
        X0 => { f(); }
        // a long non-empty block
        X1 => { some.really.long.expression.fooooooooooooooooooooooooooooooooooooooooo().baaaaarrrrrrrrrrrrrrrrrrrrrrrrrr(); }
        // an empty block
        X2 => {}
        // a short non-block
        X3 => println!("ok"),
        // a long non-block
        X4 => foo.bar.baz.test.x.y.z.a.s.d.fasdfasdf.asfads.fasd.fasdfasdf.dfasfdsaf(),
    }
}
