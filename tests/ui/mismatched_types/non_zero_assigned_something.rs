fn main() {
    let _: std::num::NonZero<u64> = 0;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`

    let _: Option<std::num::NonZero<u64>> = 0;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`

    let _: std::num::NonZero<u64> = 1;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`
    let _: std::num::NonZero<u64> = 1u8;
    //~^ ERROR mismatched types
    let _: std::num::NonZero<u64> = 1u64;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`

    let _: std::num::NonZero<u8> = 255;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`
    let _: std::num::NonZero<u8> = 256;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`

    let _: std::num::NonZero<u8> = -10;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`

    let _: std::num::NonZero<i8> = -128;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`
    let _: std::num::NonZero<i8> = -129;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`
    let _: std::num::NonZero<i8> = -1;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`
    let _: std::num::NonZero<i8> = 0;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`
    let _: std::num::NonZero<i8> = 1;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`

    let _: Option<std::num::NonZero<u64>> = 1;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`
}
