pub fn foo(params: Option<&[&str]>) -> usize {
    params.unwrap().first().unwrap().len()
}

fn main() {
    let name = "Foo";
    let x = Some(&[name]);
    let msg = foo(x);
    //~^ ERROR mismatched types
    //~| expected enum `Option<&[&str]>`
    //~| found enum `Option<&[&str; 1]>`
    //~| expected `Option<&[&str]>`, found `Option<&[&str; 1]>`
    assert_eq!(msg, 3);
}
