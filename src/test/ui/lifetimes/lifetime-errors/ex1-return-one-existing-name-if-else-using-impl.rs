// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

trait Foo {

    fn foo<'a>(x: &i32, y: &'a i32) -> &'a i32;

}

impl Foo for () {

    fn foo<'a>(x: &i32, y: &'a i32) -> &'a i32 {

        if x > y { x } else { y }
        //[base]~^ ERROR lifetime mismatch
        //[nll]~^^ ERROR lifetime may not live long enough

    }

}

fn main() {}
