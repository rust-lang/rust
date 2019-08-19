// build-pass (FIXME(62277): could be check-pass?)
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
