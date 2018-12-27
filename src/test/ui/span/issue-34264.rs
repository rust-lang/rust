fn foo(Option<i32>, String) {} //~ ERROR expected one of
//~^ ERROR expected one of
fn bar(x, y: usize) {} //~ ERROR expected one of

fn main() {
    foo(Some(42), 2);
    foo(Some(42), 2, ""); //~ ERROR this function takes
    bar("", ""); //~ ERROR mismatched types
    bar(1, 2);
    bar(1, 2, 3); //~ ERROR this function takes
}
