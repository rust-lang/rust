pub struct Value {}
pub fn new() -> Result<Value, ()> {
    Ok(Value {
    }
}
//~^ ERROR incorrect close delimiter
