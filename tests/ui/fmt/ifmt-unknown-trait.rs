fn main() {
    format!("{:notimplemented}", "3");
    //~^ ERROR: unknown format trait `notimplemented`
}
