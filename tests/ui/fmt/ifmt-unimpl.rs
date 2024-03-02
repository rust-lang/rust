fn main() {
    format!("{:X}", "3");
    //~^ ERROR trait `UpperHex` is not implemented for `str`
}
