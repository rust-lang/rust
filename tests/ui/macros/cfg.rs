fn main() {
    cfg!(); //~ ERROR macro requires a cfg-pattern
    cfg!(123); //~ ERROR malformed `cfg` macro input
    cfg!(foo = 123); //~ ERROR malformed `cfg` macro input
    cfg!(foo, bar); //~ ERROR expected 1 cfg-pattern
}
