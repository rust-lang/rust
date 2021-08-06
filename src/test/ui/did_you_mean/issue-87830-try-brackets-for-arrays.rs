fn main() {}

const FOO: [u8; 3] = { //~ ERROR this code is interpreted as a block expression
    1, 2, 3
};

const BAR: [&str; 3] = {"one", "two", "three"};
//~^ ERROR this code is interpreted as a block expression

fn foo() {
    {1, 2, 3};
    //~^ ERROR this code is interpreted as a block expression
}

fn bar() {
    1, 2, 3 //~ ERROR expected one of
}
