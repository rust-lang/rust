macro_rules! borrow {
    ($x:expr) => { &$x } //~ ERROR mismatched types
}

fn foo(_: String) {}

fn foo2(s: &String) {
    foo(s);
    //~^ ERROR mismatched types
}

fn foo3(_: u32) {}
fn foo4(u: &u32) {
    foo3(u);
    //~^ ERROR mismatched types
}

fn main() {
    let s = String::new();
    let r_s = &s;
    foo2(r_s);
    foo(&"aaa".to_owned());
    //~^ ERROR mismatched types
    foo(&mut "aaa".to_owned());
    //~^ ERROR mismatched types
    foo3(borrow!(0));
    foo4(&0);
    assert_eq!(3i32, &3i32);
    //~^ ERROR mismatched types
}
