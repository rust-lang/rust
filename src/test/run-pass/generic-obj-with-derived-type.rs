

obj handle[T](T data) {
    fn get() -> T { ret data; }
}

fn main() {
    type rgb = tup(u8, u8, u8);

    let handle[rgb] h = handle[rgb](tup(1 as u8, 2 as u8, 3 as u8));
    log "constructed object";
    log h.get()._0;
    log h.get()._1;
    log h.get()._2;
    assert (h.get()._0 == 1 as u8);
    assert (h.get()._1 == 2 as u8);
    assert (h.get()._2 == 3 as u8);
}