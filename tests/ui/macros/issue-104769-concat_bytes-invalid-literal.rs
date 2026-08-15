#![feature(concat_bytes)]

fn main() {
    concat_bytes!(7Y);
    //~^ ERROR invalid suffix `Y` for number literal
    concat_bytes!(888888888888888888888888888888888888888);
    //~^ ERROR integer literal is too large
}
