fn main() {
    // sometimes we have had trouble finding
    // the right type for f, as we unified
    // bot and u32 here
    let f = match uint::from_str(~"1234") {
        None => return (),
        Some(num) => num as u32
    };
    assert f == 1234u32;
    log(error, f)
}
