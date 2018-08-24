// Make sure that the span of try shorthand does not include the trailing
// semicolon;
fn a() -> Result<i32, ()> {
    Err(5)?; //~ ERROR 14:5: 14:12
    Ok(1)
}

fn main() {}
