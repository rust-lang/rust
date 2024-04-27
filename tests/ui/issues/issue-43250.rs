fn main() {
    let mut y;
    const C: u32 = 0;
    macro_rules! m {
        ($a:expr) => {
            let $a = 0;
        }
    }
    m!(y);
    //~^ ERROR arbitrary expressions aren't allowed in patterns
    m!(C);
    //~^ ERROR arbitrary expressions aren't allowed in patterns
}
