// rustfmt-normalize_comments: true
fn main() {
    match x {
        S(true, .., true) => (),
        S(true, ..) => (),
        S(.., true) => (),
        S(..) => (),
        S(_) => (),
        S(/* .. */ ..) => (),
        S(/* .. */ .., true) => (),
    }

    match y {
        (true, .., true) => (),
        (true, ..) => (),
        (.., true) => (),
        (..) => (),
        (_,) => (),
        (/* .. */ ..) => (),
        (/* .. */ .., true) => (),
    }
}
