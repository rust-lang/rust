// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

struct Ref<'a, T: 'a> {
    data: &'a T
}

fn foo<'a, 'b, 'c>(x: &'a mut Vec<Ref<'b, i32>>, y: Ref<'c, i32>) {
    let a: &mut Vec<Ref<i32>> = x;
    //[base]~^ ERROR lifetime mismatch
    let b = Ref { data: y.data };
    a.push(b);
    //[nll]~^ ERROR lifetime may not live long enough
}

fn main() { }
