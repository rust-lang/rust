//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:self-mapping-output.pp

#![feature(fn_delegation)]

trait Trait {
    fn method(&self) -> Self;
    fn r#static() -> Self;
    fn raw_S(&self) -> S { S }
}

struct S;
impl Trait for S {
    fn method(&self) -> S { S }
    fn r#static() -> S { S }
}

struct W(S);
impl Trait for W {
    reuse Trait::method { self.0 }
    reuse Trait::r#static;
    //~^ WARN: function cannot return without recursing [unconditional_recursion]
    reuse Trait::raw_S { self.0 }
}

impl W {
    reuse Trait::method { self.0 }
    reuse Trait::r#static;
    reuse Trait::raw_S { self.0 }
}

fn main() {}
