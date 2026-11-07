fn main() {
    cfg!(); //~ ERROR macro requires a cfg-pattern
    cfg!(123); //~ ERROR malformed `cfg` macro input
    cfg!(foo = 123); //~ ERROR malformed `cfg` macro input
    cfg!(false, false); //~ ERROR expected 1 cfg-pattern
    cfg!(foo); //~ WARN unexpected `cfg` condition name: `foo`
}
