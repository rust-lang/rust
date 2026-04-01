//@ check-pass
//
// During `Nonterminal` removal (#124141) there was at one point a problem with
// calling from_ast on expressions with inner attributes within metavars -- the
// inner attributes were being inserted in the wrong place in `from_ast`. This
// test covers that case.

macro_rules! m3 { ($e:expr) => {} }
macro_rules! m2 { ($e:expr) => { m3!($e); } }
macro_rules! m1 { ($e:expr) => { m2!($e); } }

m1!({ #![allow(unused)] 0 });

fn main() {}
