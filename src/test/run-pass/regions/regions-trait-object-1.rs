// run-pass
// This is a regression test for something that only came up while
// attempting to bootstrap libsyntax; it is adapted from
// `syntax::ext::tt::generic_extension`.


pub struct E<'a> {
    pub f: &'a u8,
}
impl<'b> E<'b> {
    pub fn m(&self) -> &'b u8 { self.f }
}

pub struct P<'c> {
    pub g: &'c u8,
}
pub trait M {
    fn n(&self) -> u8;
}
impl<'d> M for P<'d> {
    fn n(&self) -> u8 { *self.g }
}

fn extension<'e>(x: &'e E<'e>) -> Box<dyn M+'e> {
    loop {
        let p = P { g: x.m() };
        return Box::new(p) as Box<dyn M+'e>;
    }
}

fn main() {
    let w = E { f: &10 };
    let o = extension(&w);
    assert_eq!(o.n(), 10);
}
