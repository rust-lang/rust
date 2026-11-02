// Test that we can recover from very basic C arrays in the parser & provide a good diagnostic.

fn main() {}

const INTS: [u8; 3] = {
    //~^ ERROR this is a block expression, not an array
    1, 2, 3
};

const STRS: [&str; 3] = {"one", "two", "three"};
//~^ ERROR this is a block expression, not an array

fn expr_stmt() {
    {1, 2, 3};
    //~^ ERROR this is a block expression, not an array
}

// Don't trigger here.
fn unsafe_block() {
    unsafe { 1, 2, 3 } //~ ERROR expected one of
}

// Don't trigger here.
fn labeled_block() {
    'label: { 1, 2, 3 } //~ ERROR expected one of
}

// Don't trigger here, this is not a block expression, only a block.
fn fn_body_block() {
    1, 2, 3 //~ ERROR expected one of
}

// Don't trigger here, this is not a block expression, only a block.
fn closure_body_block() {
    || -> i32 { 1, 2, 3 }; //~ ERROR expected one of
}

// Don't trigger here.
fn const_arg() {
    struct Casket<const N: usize>;
    Casket::<{ 1, 2, 3 }>; //~ ERROR expected one of
}
