fn foo() -> i32 { //~ HELP otherwise consider changing the return type to account for that possibility
    for i in 0..0 { //~ ERROR mismatched types [E0308]
        return i;
    } //~ HELP return a value for the case when the loop has zero elements to iterate on
}

fn main() {}
