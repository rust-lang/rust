// run-rustfix

#![allow(unused_must_use, unused_comparisons)]

macro_rules! is_plainly_printable {
    ($i: ident) => {
        ($i as u32) < 0 //~ `<` is interpreted as a start of generic arguments
    };
}

fn main() {
    let c = 'a';
    is_plainly_printable!(c);
}
