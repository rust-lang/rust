struct S;

impl Iterator for S {
    type Item = i32;
    fn next(&mut self) -> Result<i32, i32> { Ok(7) }
    //~^ ERROR method `next` has an incompatible type for trait
    //~| expected enum `std::option::Option`, found enum `std::result::Result`
}

fn main() {}
