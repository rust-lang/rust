// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

trait Dummy { fn dummy(&self); }

fn foo1<'a:'b,'b>(x: &'a mut (dyn Dummy+'a)) -> &'b mut (dyn Dummy+'b) {
    // Here, we are able to coerce
    x
}

fn foo2<'a:'b,'b>(x: &'b mut (dyn Dummy+'a)) -> &'b mut (dyn Dummy+'b) {
    // Here, we are able to coerce
    x
}

fn foo3<'a,'b>(x: &'a mut dyn Dummy) -> &'b mut dyn Dummy {
    // Without knowing 'a:'b, we can't coerce
    x
    //[base]~^ ERROR lifetime bound not satisfied
    //[base]~| ERROR cannot infer an appropriate lifetime
    //[nll]~^^^ ERROR lifetime may not live long enough
}

struct Wrapper<T>(T);
fn foo4<'a:'b,'b>(x: Wrapper<&'a mut dyn Dummy>) -> Wrapper<&'b mut dyn Dummy> {
    // We can't coerce because it is packed in `Wrapper`
    x
    //[base]~^ ERROR mismatched types
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn main() {}
