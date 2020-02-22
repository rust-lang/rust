fn main() {}

extern {
    pub pub fn foo();
    //~^ ERROR unmatched visibility `pub`
    //~| ERROR non-item in item list
}
