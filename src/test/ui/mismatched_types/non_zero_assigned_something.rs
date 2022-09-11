fn main() {
    let _: std::num::NonZeroU64 = 1;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZeroU64::new`

    let _: Option<std::num::NonZeroU64> = 1;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZeroU64::new`
}
