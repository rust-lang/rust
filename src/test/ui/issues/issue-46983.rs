// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

fn foo(x: &u32) -> &'static u32 {
    &*x
    //[base]~^ ERROR `x` has an anonymous lifetime `'_` but it needs to satisfy a `'static` lifetime requirement [E0759]
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn main() {}
