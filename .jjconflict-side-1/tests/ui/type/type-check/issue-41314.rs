enum X {
    Y(u32)
}

fn main() {
    match X::Y(0) {
        X::Y { number } => {}
        //~^ ERROR tuple variant `X::Y` written as struct variant
    }
}
