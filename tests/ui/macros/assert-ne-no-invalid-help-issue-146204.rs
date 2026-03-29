fn main() {
    let buf = [0_u8; 4];
    assert_ne!(buf, b"----");
    //~^ ERROR can't compare `[u8; 4]` with `&[u8; 4]`

    assert_eq!(buf, b"----");
    //~^ ERROR can't compare `[u8; 4]` with `&[u8; 4]`
}
