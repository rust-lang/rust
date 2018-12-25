pub fn foo(params: Option<&[&str]>) -> usize {
    params.unwrap().first().unwrap().len()
}

fn main() {
    let name = "Foo";
    let x = Some(&[name]);
    let msg = foo(x);
//~^ ERROR mismatched types
//~| expected type `std::option::Option<&[&str]>`
//~| found type `std::option::Option<&[&str; 1]>`
//~| expected slice, found array of 1 elements
    assert_eq!(msg, 3);
}
