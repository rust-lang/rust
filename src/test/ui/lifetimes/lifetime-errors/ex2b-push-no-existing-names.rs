// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

struct Ref<'a, T: 'a> {
    data: &'a T
}

fn foo(x: &mut Vec<Ref<i32>>, y: Ref<i32>) {
    x.push(y);
    //[base]~^ ERROR lifetime mismatch
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn main() { }
