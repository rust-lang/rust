impl S {
    fn a(self) {}
    fn b(&self,) {}
    fn c(&'a self,) {}
    fn d(&'a mut self, x: i32) {}
    fn e(mut self) {}
}
