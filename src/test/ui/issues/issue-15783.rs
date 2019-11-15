pub fn foo(params: Option<&[&str]>) -> usize {
    params.unwrap().first().unwrap().len()
}

fn main() {
    let name = "Foo";
    let x = Some(&[name]);
    let msg = foo(x);
    //~^ ERROR mismatched types
    //~| expected enum `std::option::Option<&[&str]>`
    //~| found enum `std::option::Option<&[&str; 1]>`
    //~| expected slice `[&str]`, found array `[&str; 1]`
    assert_eq!(msg, 3);
}
