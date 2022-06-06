// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

struct Ref<'a> {
    x: &'a u32,
}

fn foo(mut x: Vec<Ref>, y: Ref) {
    x.push(y);
    //[base]~^ ERROR lifetime mismatch
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn main() {}
