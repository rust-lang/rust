struct Thing;

trait Method<T> {
    fn method(&self) -> T;
    fn mut_method(&mut self) -> T;
}

impl Method<i32> for Thing {
    fn method(&self) -> i32 { 0 }
    fn mut_method(&mut self) -> i32 { 0 }
}

impl Method<u32> for Thing {
    fn method(&self) -> u32 { 0 }
    fn mut_method(&mut self) -> u32 { 0 }
}

fn main() {
    let thing = Thing;
    thing.method();
    //~^ ERROR type annotations needed
    //~| ERROR type annotations needed
    thing.mut_method(); //~ ERROR type annotations needed
}
