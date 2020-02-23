fn main() {}

extern {
    pub pub fn foo();
    //~^ ERROR visibility `pub` not followed by an item
    //~| ERROR non-item in item list
}
