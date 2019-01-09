fn main() {
    let x = 0;
    match 1 {
        0 ..= x => {}
        //~^ ERROR runtime values cannot be referenced in patterns
    };
}
