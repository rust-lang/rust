// rustfmt-normalize_comments: true
// rustfmt-condense_wildcard_suffices: true

fn main() {
    match x {
        Butt(..) => "hah",
        Tup(_) => "nah",
        Quad(_, _, x, _) => " also no rewrite",
        Quad(x, ..) => "condense me pls",
        Weird(x,
              _,
              _,
              // dont condense before
              ..) => "pls work",
    }
}
