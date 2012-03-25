fn main() {
    // sometimes we have had trouble finding
    // the right type for f, as we unified
    // bot and u32 here
    let f = alt uint::from_str("1234") {
        none { ret () }
        some(num) { num as u32 }
    };
    assert f == 1234u32;
    log(error, f)
}
