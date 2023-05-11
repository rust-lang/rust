// compile-flags: --edition 2018
fn foo() -> Result<(), ()> {
    Ok(try!()); //~ ERROR use of deprecated `try` macro
    Ok(try!(Ok(()))) //~ ERROR use of deprecated `try` macro
}

fn main() {
    let _ = foo();
}
