// rust-lang/rust#52059: Regardless of whether you are moving out of a
// Drop type or just introducing an inadvertent alias via a borrow of
// one of its fields, it is useful to be reminded of the significance
// of the fact that the type implements Drop.

pub struct S<'a> { url: &'a mut String }

impl<'a> Drop for S<'a> { fn drop(&mut self) { } }

fn finish_1(s: S) -> &mut String {
    s.url
}
//~^^ ERROR borrow may still be in use when destructor runs

fn finish_2(s: S) -> &mut String {
    let p = &mut *s.url; p
}
//~^^ ERROR borrow may still be in use when destructor runs

fn finish_3(s: S) -> &mut String {
    let p: &mut _ = s.url; p
}
//~^^ ERROR borrow may still be in use when destructor runs

fn finish_4(s: S) -> &mut String {
    let p = s.url; p
}
//~^^ ERROR cannot move out of type `S<'_>`, which implements the `Drop` trait

fn main() {}
