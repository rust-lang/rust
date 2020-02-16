fn main() {
    let x = 255u8;
    let 0u8..=x = 0;
    //~^ ERROR runtime values cannot be referenced in patterns
}
