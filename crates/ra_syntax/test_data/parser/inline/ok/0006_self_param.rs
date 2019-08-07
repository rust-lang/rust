impl S {
    fn a(self) {}
    fn b(&self,) {}
    fn c(&'a self,) {}
    fn d(&'a mut self, x: i32) {}
    fn e(mut self) {}
    fn f(#[must_use] self) {}
    fn g1(#[attr] self) {}
    fn g2(#[attr] &self) {}
    fn g3<'a>(#[attr] &mut self) {}
    fn g4<'a>(#[attr] &'a self) {}
    fn g5<'a>(#[attr] &'a mut self) {}
}
