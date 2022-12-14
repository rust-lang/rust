// We didn't have a single test mentioning
// `ReEmpty` and this test changes that.
fn foo<'a>(_a: &'a u32) where for<'b> &'b (): 'a {
}

fn main() {
    foo(&10);
    //~^ ERROR higher-ranked lifetime error
    //~| NOTE could not prove
}
