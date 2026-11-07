fn g1(#[attr1] #[attr2] pat: Type) {}
fn g2(#[attr1] x: u8) {}

extern "C" { fn printf(format: *const i8, #[attr] ...) -> i32; }

trait Foo {
    fn bar(#[attr] _: u64, # [attr] mut x: i32);
}

impl S {
     fn f(#[must_use] self) {}
     fn g1(#[attr] self) {}
     fn g2(#[attr] &self) {}
     fn g3<'a>(#[attr] &mut self) {}
     fn g4<'a>(#[attr] &'a self) {}
     fn g5<'a>(#[attr] &'a mut self) {}
     fn c(#[attr] self: Self) {}
     fn d(#[attr] self: Rc<Self>) {}
}