// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

// We didn't have a single test mentioning
// `ReEmpty` and this test changes that.
fn foo<'a>(_a: &'a u32) where for<'b> &'b (): 'a {
    //[base]~^ NOTE type must outlive the empty lifetime as required by this binding
}

fn main() {
    foo(&10);
    //[base]~^ ERROR the type `&'b ()` does not fulfill the required lifetime
    //[nll]~^^ ERROR higher-ranked lifetime error
    //[nll]~| NOTE could not prove
}
