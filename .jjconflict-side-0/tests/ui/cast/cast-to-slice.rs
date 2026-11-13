fn main() {
    "example".as_bytes() as [char];
    //~^ ERROR cast to unsized type

    let arr: &[u8] = &[0, 2, 3];
    arr as [char];
    //~^ ERROR cast to unsized type
}
