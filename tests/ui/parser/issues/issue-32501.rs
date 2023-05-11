fn main() {
    let a = 0;
    let _b = 0;
    let _ = 0;
    let mut b = 0;
    let mut _b = 0;
    let mut _ = 0;
    //~^ ERROR `mut` must be followed by a named binding
}
