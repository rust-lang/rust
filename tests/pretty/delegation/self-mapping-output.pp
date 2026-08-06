#![attr = Feature([fn_delegation#0])]
extern crate std;
#[attr = PreludeImport]
use ::std::prelude::rust_2015::*;
//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:self-mapping-output.pp


trait Trait {
    fn method(&self)
    -> Self;
    fn r#static()
    -> Self;
    fn raw_S(&self) -> S { S }
}

struct S;
impl Trait for S {
    fn method(&self) -> S { S }
    fn r#static() -> S { S }
}

struct W(S);
impl Trait for W {
    #[attr = Inline(Hint)]
    fn method(self: _) -> _ { Self { 0: Trait::method(self.0) } }
    #[attr = Inline(Hint)]
    fn r#static() -> _ { Trait::r#static() }
    //~^ WARN: function cannot return without recursing [unconditional_recursion]
    #[attr = Inline(Hint)]
    fn raw_S(self: _) -> _ { Trait::raw_S(self.0) }
}

impl W {
    #[attr = Inline(Hint)]
    fn method(self: _) -> _ { Self { 0: Trait::method(self.0) } }
    #[attr = Inline(Hint)]
    fn r#static() -> _ { Trait::r#static() }
    #[attr = Inline(Hint)]
    fn raw_S(self: _) -> _ { Trait::raw_S(self.0) }
}

fn main() { }
