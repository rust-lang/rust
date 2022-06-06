// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

trait Foo {
    fn foo<'a>(x: &mut Vec<&u8>, y: &u8);
}
impl Foo for () {
    fn foo(x: &mut Vec<&u8>, y: &u8) {
        x.push(y);
        //[base]~^ ERROR lifetime mismatch
        //[nll]~^^ ERROR lifetime may not live long enough
    }
}
fn main() {}
