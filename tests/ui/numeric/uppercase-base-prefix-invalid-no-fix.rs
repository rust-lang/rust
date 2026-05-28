// Checks that integers with seeming uppercase base prefixes do not get bogus capitalization
// suggestions.

fn main() {
    _ = 123X1a3;
    //~^ ERROR invalid suffix `X1a3` for number literal
    //~| NOTE invalid suffix `X1a3`
    //~| HELP the suffix must be one of the numeric types (`u32`, `isize`, `f32`, etc.)

    _ = 456O123;
    //~^ ERROR invalid suffix `O123` for number literal
    //~| NOTE invalid suffix `O123`
    //~| HELP the suffix must be one of the numeric types (`u32`, `isize`, `f32`, etc.)

    _ = 789B101;
    //~^ ERROR invalid suffix `B101` for number literal
    //~| NOTE invalid suffix `B101`
    //~| HELP the suffix must be one of the numeric types (`u32`, `isize`, `f32`, etc.)

    _ = 0XYZ;
    //~^ ERROR invalid suffix `XYZ` for number literal
    //~| NOTE invalid suffix `XYZ`
    //~| HELP the suffix must be one of the numeric types (`u32`, `isize`, `f32`, etc.)

    _ = 0OPQ;
    //~^ ERROR invalid suffix `OPQ` for number literal
    //~| NOTE invalid suffix `OPQ`
    //~| HELP the suffix must be one of the numeric types (`u32`, `isize`, `f32`, etc.)

    _ = 0BCD;
    //~^ ERROR invalid suffix `BCD` for number literal
    //~| NOTE invalid suffix `BCD`
    //~| HELP the suffix must be one of the numeric types (`u32`, `isize`, `f32`, etc.)
}
