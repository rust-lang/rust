// Make sure that the span of try shorthand does not include the trailing
// semicolon;
fn a() -> Result<i32, ()> {
    Err(5)?; //~ ERROR
    Ok(1)
}

fn main() {}
