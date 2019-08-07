impl S {
    fn a(self: &Self) {}
    fn b(mut self: Box<Self>) {}
    fn c(#[attr] self: Self) {}
    fn d(#[attr] self: Rc<Self>) {}
}
