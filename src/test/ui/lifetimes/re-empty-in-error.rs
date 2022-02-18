// We didn't have a single test mentioning
// `ReEmpty` and this test changes that.
fn foo<'a>(_a: &'a u32) where for<'b> &'b (): 'a {
    //~^ NOTE type must outlive the empty lifetime as required by this binding
}

fn main() {
    foo(&10);
    //~^ ERROR the type `&'b ()` does not fulfill the required lifetime
}
