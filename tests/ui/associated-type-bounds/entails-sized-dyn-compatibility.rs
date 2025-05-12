//@ build-pass (FIXME(62277): could be check-pass?)

trait Tr1: Sized { type As1; }
trait Tr2<'a>: Sized { type As2; }

trait ObjTr1 { fn foo() -> Self where Self: Tr1<As1: Copy>; }
fn _assert_dyn_compat_1(_: Box<dyn ObjTr1>) {}

trait ObjTr2 { fn foo() -> Self where Self: Tr1<As1: 'static>; }
fn _assert_dyn_compat_2(_: Box<dyn ObjTr2>) {}

trait ObjTr3 { fn foo() -> Self where Self: Tr1<As1: Into<u8> + 'static + Copy>; }
fn _assert_dyn_compat_3(_: Box<dyn ObjTr3>) {}

trait ObjTr4 { fn foo() -> Self where Self: Tr1<As1: for<'a> Tr2<'a>>; }
fn _assert_dyn_compat_4(_: Box<dyn ObjTr4>) {}

trait ObjTr5 { fn foo() -> Self where for<'a> Self: Tr1<As1: Tr2<'a>>; }
fn _assert_dyn_compat_5(_: Box<dyn ObjTr5>) {}

trait ObjTr6 { fn foo() -> Self where Self: for<'a> Tr1<As1: Tr2<'a, As2: for<'b> Tr2<'b>>>; }
fn _assert_dyn_compat_6(_: Box<dyn ObjTr6>) {}

fn main() {}
