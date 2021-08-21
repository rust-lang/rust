#![crate_type = "lib"]
enum Enum {
    //~^ ERROR `#[repr(inttype)]` must be specified
    Unit = 5,
    Tuple(u8) = 3,
    Struct {
        foo: u16
    } = 1,
}
