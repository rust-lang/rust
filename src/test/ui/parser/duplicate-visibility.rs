fn main() {}

extern "C" {
    pub pub fn foo();
    //~^ ERROR visibility `pub` is not followed by an item
    //~| ERROR non-item in item list
}
