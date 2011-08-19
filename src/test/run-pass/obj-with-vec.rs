

fn main() {
    obj buf(data: [u8]) {
        fn get(i: int) -> u8 { ret data[i]; }
    }
    let b = buf([1 as u8, 2 as u8, 3 as u8]);
    log b.get(1);
    assert (b.get(1) == 2 as u8);
}
