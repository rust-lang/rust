//@ check-pass
// Tests that we properly lint at 'paren' expressions

fn foo() -> Result<(), String>  {
    (try!(Ok::<u8, String>(1))); //~ WARN use of deprecated macro `try`
    Ok(())
}

fn main() {}
