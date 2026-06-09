fn main() {
    let _: std::num::NonZero<u64> = 1;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`

    let _: Option<std::num::NonZero<u64>> = 1;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`
}
