fn main() {
    0b101010f64;
    //~^ ERROR binary float literal is not supported
    0b101.010;
    //~^ ERROR binary float literal is not supported
    0b101p4f64;
    //~^ ERROR invalid suffix `p4f64` for number literal
}
