// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

struct Ref<'a, 'b> { a: &'a u32, b: &'b u32 }

fn foo(mut y: Ref, x: &u32) {
    y.b = x;
    //[base]~^ ERROR lifetime mismatch
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn main() { }
