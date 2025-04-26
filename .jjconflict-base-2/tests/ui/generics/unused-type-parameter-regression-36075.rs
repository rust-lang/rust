// https://github.com/rust-lang/rust/issues/36075
//@ check-pass
#![allow(dead_code)]
trait DeclarationParser {
    type Declaration;
}

struct DeclarationListParser<'i, I, P>
    where P: DeclarationParser<Declaration = I>
{
    input: &'i (),
    parser: P
}

fn main() {}
