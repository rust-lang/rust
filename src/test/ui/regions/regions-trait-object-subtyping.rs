trait Dummy { fn dummy(&self); }

fn foo1<'a:'b,'b>(x: &'a mut (Dummy+'a)) -> &'b mut (Dummy+'b) {
    // Here, we are able to coerce
    x
}

fn foo2<'a:'b,'b>(x: &'b mut (Dummy+'a)) -> &'b mut (Dummy+'b) {
    // Here, we are able to coerce
    x
}

fn foo3<'a,'b>(x: &'a mut Dummy) -> &'b mut Dummy {
    // Without knowing 'a:'b, we can't coerce
    x //~ ERROR lifetime bound not satisfied
     //~^ ERROR cannot infer an appropriate lifetime
}

struct Wrapper<T>(T);
fn foo4<'a:'b,'b>(x: Wrapper<&'a mut Dummy>) -> Wrapper<&'b mut Dummy> {
    // We can't coerce because it is packed in `Wrapper`
    x //~ ERROR mismatched types
}

fn main() {}
