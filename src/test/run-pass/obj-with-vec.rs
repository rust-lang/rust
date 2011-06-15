

fn main() {
    obj buf(vec[u8] data) {
        fn get(int i) -> u8 { ret data.(i); }
    }
    auto b = buf([1 as u8, 2 as u8, 3 as u8]);
    log b.get(1);
    assert (b.get(1) == 2 as u8);
}