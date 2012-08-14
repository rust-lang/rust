// Here we are checking that a reasonable error msg is provided.
//
// The current message is not ideal, but we used to say "borrowed
// pointer has lifetime &, but the borrowed value only has lifetime &"
// which is definitely no good.


fn get() -> &int {
    let x = 3;
    return &x;
    //~^ ERROR illegal borrow: borrowed pointer must be valid for the lifetime & as defined on the block at 8:17, but the borrowed value is only valid for the block at 8:17
}

fn main() {}
