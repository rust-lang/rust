// Here we are checking that a reasonable error msg is provided.
//
// The current message is not ideal, but we used to say "borrowed
// pointer has lifetime &, but the borrowed value only has lifetime &"
// which is definitely no good.


fn get() -> &int {
    //~^ NOTE borrowed pointer must be valid for the anonymous lifetime #1 defined on
    //~^^ NOTE ...but borrowed value is only valid for the block at
    let x = 3;
    return &x;
    //~^ ERROR illegal borrow
}

fn main() {}
