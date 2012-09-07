fn get_third<T: Copy>(t: (T, T, T)) -> T { let (_, _, x) = t; return x; }

fn main() {
    log(debug, get_third((1, 2, 3)));
    assert (get_third((1, 2, 3)) == 3);
    assert (get_third((5u8, 6u8, 7u8)) == 7u8);
}
