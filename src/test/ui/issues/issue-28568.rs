struct MyStruct;

impl Drop for MyStruct {
    fn drop(&mut self) { }
}

impl Drop for MyStruct {
//~^ ERROR conflicting implementations of trait
    fn drop(&mut self) { }
}

fn main() {}
