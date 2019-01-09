enum X {
    Y(u32)
}

fn main() {
    match X::Y(0) {
        X::Y { number } => {} //~ ERROR does not have a field named `number`
        //~^ ERROR pattern does not mention field `0`
    }
}
