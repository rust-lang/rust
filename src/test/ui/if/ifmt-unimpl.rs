fn main() {
    format!("{:X}", "3");
    //~^ ERROR: `str: std::fmt::UpperHex` is not satisfied
}
