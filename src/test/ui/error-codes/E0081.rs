enum Enum {
    P = 3,
    //~^ NOTE first use of `3`
    X = 3,
    //~^ ERROR discriminant value `3` already exists
    //~| NOTE enum already has `3`
    Y = 5
}

#[repr(u8)]
enum EnumOverflowRepr {
    P = 257,
    //~^ NOTE first use of `1` (overflowed from `257`)
    X = 513,
    //~^ ERROR discriminant value `1` already exists
    //~| NOTE enum already has `1` (overflowed from `513`)
}

fn main() {
}
