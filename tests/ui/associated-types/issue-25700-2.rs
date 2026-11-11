//@ run-pass
pub trait Parser {
    type Input;
}

pub struct Iter<P: Parser>(#[allow(dead_code)] P, P::Input);

#[allow(dead_code)]
pub struct Map<P, F>(P, F);
impl<P, F> Parser for Map<P, F> where F: FnMut(P) {
    type Input = u8;
}

trait AstId { type Untyped; }
impl AstId for u32 { type Untyped = u32; }

fn record_type<Id: AstId>(i: Id::Untyped) -> u8 {
    Iter(Map(i, |_: Id::Untyped| {}), 42).1
}

pub fn main() {
    assert_eq!(record_type::<u32>(3), 42);
}
