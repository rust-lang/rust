fn main() {}

const FOO: [u8; 3] = {
    //~^ ERROR this is a block expression, not an array
    1, 2, 3
};

const BAR: [&str; 3] = {"one", "two", "three"};
//~^ ERROR this is a block expression, not an array

fn foo() {
    {1, 2, 3};
    //~^ ERROR this is a block expression, not an array
}

fn bar() {
    1, 2, 3 //~ ERROR expected one of
}
