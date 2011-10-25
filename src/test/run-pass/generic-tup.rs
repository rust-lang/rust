fn get_third<T>(t: (T, T, T)) -> T { let (_, _, x) = t; ret x; }

fn main() {
    log get_third((1, 2, 3));
    assert (get_third((1, 2, 3)) == 3);
    assert (get_third((5u8, 6u8, 7u8)) == 7u8);
}
