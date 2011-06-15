

fn get_third[T](&tup(T, T, T) t) -> T { ret t._2; }

fn main() {
    log get_third(tup(1, 2, 3));
    assert (get_third(tup(1, 2, 3)) == 3);
    assert (get_third(tup(5u8, 6u8, 7u8)) == 7u8);
}