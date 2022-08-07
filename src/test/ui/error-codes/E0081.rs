enum Enum {
    //~^ ERROR discriminant value `3` assigned more than once
    P = 3,
    //~^ NOTE `3` assigned here
    X = 3,
    //~^ NOTE `3` assigned here
    Y = 5
}

#[repr(u8)]
enum EnumOverflowRepr {
    //~^ ERROR discriminant value `1` assigned more than once
    P = 257,
    //~^ NOTE `1` (overflowed from `257`) assigned here
    X = 513,
    //~^ NOTE `1` (overflowed from `513`) assigned here
}

#[repr(i8)]
enum NegDisEnum {
    //~^ ERROR discriminant value `-1` assigned more than once
    First = -1,
    //~^ NOTE `-1` assigned here
    Second = -2,
    //~^ NOTE discriminant for `Last` incremented from this startpoint (`Second` + 1 variant later => `Last` = -1)
    Last,
    //~^ NOTE `-1` assigned here
}

#[repr(i32)]
enum MultipleDuplicates {
    //~^ ERROR discriminant value `0` assigned more than once
    V0,
    //~^ NOTE `0` assigned here
    V1 = 0,
    //~^ NOTE `0` assigned here
    V2,
    V3,
    V4 = 0,
    //~^ NOTE `0` assigned here
    V5 = -2,
    //~^ NOTE discriminant for `V7` incremented from this startpoint (`V5` + 2 variant later => `V7` = 0)
    V6,
    V7,
    //~^ NOTE `0` assigned here
}

fn main() {
}
