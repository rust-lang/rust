enum Enum {
    //~^ ERROR discriminant value `3` assigned more than once
    P = 3,
    //~^ NOTE first assignment of `3`
    X = 3,
    //~^ NOTE second assignment of `3`
    Y = 5
}

#[repr(u8)]
enum EnumOverflowRepr {
    //~^ ERROR discriminant value `1` assigned more than once
    P = 257,
    //~^ NOTE first assignment of `1` (overflowed from `257`)
    X = 513,
    //~^ NOTE second assignment of `1` (overflowed from `513`)
}

#[repr(i8)]
enum NegDisEnum {
    //~^ ERROR discriminant value `-1` assigned more than once
    First = -1,
    //~^ NOTE first assignment of `-1`
    Second = -2,
    //~^ NOTE assigned discriminant for `Last` was incremented from this discriminant
    Last,
    //~^ NOTE second assignment of `-1`
}

fn main() {
}
