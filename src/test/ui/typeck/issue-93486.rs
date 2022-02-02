fn main() {
    while let 1 = 1 {
        vec![].last_mut().unwrap() = 3_u8;
        //~^ ERROR invalid left-hand side of assignment
    }
}
