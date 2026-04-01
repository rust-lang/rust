fn main() {
    0xABC.Df;
    //~^ ERROR `{integer}` is a primitive type and therefore doesn't have fields
    0x567.89;
    //~^ ERROR hexadecimal float literal is not supported
    0xDEAD.BEEFp-2f;
    //~^ ERROR invalid suffix `f` for float literal
    //~| ERROR `{integer}` is a primitive type and therefore doesn't have fields
}
