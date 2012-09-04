enum chan { chan_t, }

impl chan : cmp::Eq {
    pure fn eq(&&other: chan) -> bool {
        (self as uint) == (other as uint)
    }
}

fn wrapper3(i: chan) {
    assert i == chan_t;
}

fn main() {
    let wrapped = {||wrapper3(chan_t)};
    wrapped();
}
