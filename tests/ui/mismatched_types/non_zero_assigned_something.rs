fn main() {
    let _: std::num::NonZero<u64> = 0;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`

    let _: Option<std::num::NonZero<u64>> = 0;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`

    let _: std::num::NonZero<u64> = 0;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`

    let _: Option<std::num::NonZero<u64>> = 0;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`

    let _: std::num::NonZero<u64> = 1;
    let _: std::num::NonZero<u64> = 1u8;
    //~^ ERROR mismatched types
    let _: std::num::NonZero<u64> = 1u64;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`

    let _: std::num::NonZero<u8> = 255;
    let _: std::num::NonZero<u8> = 256; // errors only if no other errors occurred

    let _: std::num::NonZero<u8> = -10;
    //~^ ERROR cannot apply unary operator `-` to type `NonZero<u8>`

    let _: std::num::NonZero<i8> = -128;
    let _: std::num::NonZero<i8> = -129; // errors only if no other errors occurred
    let _: std::num::NonZero<i8> = -1;
    let _: std::num::NonZero<i8> = 0;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`
    let _: std::num::NonZero<i8> = 1;

    let _: Option<std::num::NonZero<u64>> = 1;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`
}
