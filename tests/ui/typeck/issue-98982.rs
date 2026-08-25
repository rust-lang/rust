fn foo() -> i32 {
    for i in 0..0 { //~ ERROR mismatched types [E0308]
        return i;
    } //~ HELP consider returning a value here
}

fn main() {}
