

obj handle<copy T>(data: T) {
    fn get() -> T { ret data; }
}

fn main() {
    type rgb = {x: u8, y: u8, z: u8};

    let h: handle<rgb> = handle::<rgb>({x: 1 as u8, y: 2 as u8, z: 3 as u8});
    #debug("constructed object");
    log_full(core::debug, h.get().x);
    log_full(core::debug, h.get().y);
    log_full(core::debug, h.get().z);
    assert (h.get().x == 1 as u8);
    assert (h.get().y == 2 as u8);
    assert (h.get().z == 3 as u8);
}
