fn main() {}

pub struct Value {}
pub fn new() -> Result<Value, ()> {
    Ok(Value {
    }
}
//~^ ERROR mismatched closing delimiter
