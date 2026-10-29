//@ run-rustfix
// Checks that integers with an uppercase base prefix (0B, 0X, 0O) have a nice error
#![allow(unused_variables)]

fn main() {
    let a = 0XABCDEF;
    //~^ ERROR invalid base prefix for number literal
    //~| NOTE base prefixes (`0xff`, `0b1010`, `0o755`) are lowercase
    //~| HELP try making the prefix lowercase
    //~| SUGGESTION 0xABCDEF

    let b = 0O755;
    //~^ ERROR invalid base prefix for number literal
    //~| NOTE base prefixes (`0xff`, `0b1010`, `0o755`) are lowercase
    //~| HELP try making the prefix lowercase
    //~| SUGGESTION 0o755

    let c = 0B10101010;
    //~^ ERROR invalid base prefix for number literal
    //~| NOTE base prefixes (`0xff`, `0b1010`, `0o755`) are lowercase
    //~| HELP try making the prefix lowercase
    //~| SUGGESTION 0b10101010

    let d = 0XABC_DEF;
    //~^ ERROR invalid base prefix for number literal
    //~| NOTE base prefixes (`0xff`, `0b1010`, `0o755`) are lowercase
    //~| HELP try making the prefix lowercase
    //~| SUGGESTION 0xABC_DEF

    let e = 0O7_55;
    //~^ ERROR invalid base prefix for number literal
    //~| NOTE base prefixes (`0xff`, `0b1010`, `0o755`) are lowercase
    //~| HELP try making the prefix lowercase
    //~| SUGGESTION 0o7_55

    let f = 0B1010_1010;
    //~^ ERROR invalid base prefix for number literal
    //~| NOTE base prefixes (`0xff`, `0b1010`, `0o755`) are lowercase
    //~| HELP try making the prefix lowercase
    //~| SUGGESTION 0b1010_1010

    let g = 0XABC_DEF_u64;
    //~^ ERROR invalid base prefix for number literal
    //~| NOTE base prefixes (`0xff`, `0b1010`, `0o755`) are lowercase
    //~| HELP try making the prefix lowercase
    //~| SUGGESTION 0xABC_DEF_u64

    let h = 0O7_55_u32;
    //~^ ERROR invalid base prefix for number literal
    //~| NOTE base prefixes (`0xff`, `0b1010`, `0o755`) are lowercase
    //~| HELP try making the prefix lowercase
    //~| SUGGESTION 0o7_55_u32

    let i = 0B1010_1010_u8;
    //~^ ERROR invalid base prefix for number literal
    //~| NOTE base prefixes (`0xff`, `0b1010`, `0o755`) are lowercase
    //~| HELP try making the prefix lowercase
    //~| SUGGESTION 0b1010_1010_u8
    //
    let j = 0XABCDEFu64;
    //~^ ERROR invalid base prefix for number literal
    //~| NOTE base prefixes (`0xff`, `0b1010`, `0o755`) are lowercase
    //~| HELP try making the prefix lowercase
    //~| SUGGESTION 0xABCDEFu64

    let k = 0O755u32;
    //~^ ERROR invalid base prefix for number literal
    //~| NOTE base prefixes (`0xff`, `0b1010`, `0o755`) are lowercase
    //~| HELP try making the prefix lowercase
    //~| SUGGESTION 0o755u32

    let l = 0B10101010u8;
    //~^ ERROR invalid base prefix for number literal
    //~| NOTE base prefixes (`0xff`, `0b1010`, `0o755`) are lowercase
    //~| HELP try making the prefix lowercase
    //~| SUGGESTION 0b10101010u8
}
