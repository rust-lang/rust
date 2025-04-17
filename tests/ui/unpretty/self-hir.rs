//@ compile-flags: -Zunpretty=hir
//@ check-pass
//@ edition: 2015

pub struct Bar {
    a: String,
    b: u8,
}

impl Bar {
    fn imm_self(self) {}
    fn mut_self(mut self) {}
    fn refimm_self(&self) {}
    fn refmut_self(&mut self) {}
}
