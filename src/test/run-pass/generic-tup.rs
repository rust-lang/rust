fn get_third<copy T>(t: (T, T, T)) -> T { let (_, _, x) = t; ret x; }

fn main() {
    log_full(core::debug, get_third((1, 2, 3)));
    assert (get_third((1, 2, 3)) == 3);
    assert (get_third((5u8, 6u8, 7u8)) == 7u8);
}
