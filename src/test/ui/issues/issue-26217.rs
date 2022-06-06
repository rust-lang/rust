// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

fn foo<T>() where for<'a> T: 'a {}

fn bar<'a>() {
    foo::<&'a i32>();
    //[base]~^ ERROR the type `&'a i32` does not fulfill the required lifetime
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn main() {
    bar();
}
