

obj handle[T](T data) {
    fn get() -> T { ret data; }
}

fn main() {
    type rgb = rec(u8 x, u8 y, u8 z);

    let handle[rgb] h = handle[rgb](rec(x=1 as u8,
                                        y=2 as u8,
                                        z=3 as u8));
    log "constructed object";
    log h.get().x;
    log h.get().y;
    log h.get().z;
    assert (h.get().x == 1 as u8);
    assert (h.get().y == 2 as u8);
    assert (h.get().z == 3 as u8);
}