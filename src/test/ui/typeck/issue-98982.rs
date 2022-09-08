fn foo() -> i32 {
    for i in 0..0 {
    //~^ ERROR: mismatched types [E0308]
        return i;
    }
    //~| help: return a value for the case when the loop has zero elements to iterate on, or consider changing the return type to account for that possibility
}

fn main() {}
