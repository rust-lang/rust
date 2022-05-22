// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

fn foo(_: impl for<'a> FnOnce(&'a u32, &u32) -> &'a u32) {
}

fn main() {
    foo(|a, b| b)
    //[base]~^ ERROR lifetime of reference outlives lifetime of borrowed content...
    //[nll]~^^ ERROR lifetime may not live long enough
}
