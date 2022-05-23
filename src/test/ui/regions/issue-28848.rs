// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

struct Foo<'a, 'b: 'a>(&'a &'b ());

impl<'a, 'b> Foo<'a, 'b> {
    fn xmute(a: &'b ()) -> &'a () {
        unreachable!()
    }
}

pub fn foo<'a, 'b>(u: &'b ()) -> &'a () {
    Foo::<'a, 'b>::xmute(u)
    //[base]~^ ERROR lifetime bound not satisfied
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn main() {}
