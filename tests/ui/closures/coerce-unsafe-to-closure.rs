fn main() {
    let x: Option<&[u8]> = Some("foo").map(std::mem::transmute);
    //~^ ERROR E0277
}
