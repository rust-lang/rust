fn main() {
    match 5u32 {
        1000 ..= 5 => {}
        //~^ ERROR lower bound for range pattern must be less than or equal to upper bound
    }
}
