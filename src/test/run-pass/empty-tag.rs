enum chan { chan_t, }

impl chan : cmp::Eq {
    pure fn eq(&self, other: &chan) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    pure fn ne(&self, other: &chan) -> bool { !(*self).eq(other) }
}

fn wrapper3(i: chan) {
    assert i == chan_t;
}

fn main() {
    let wrapped = {||wrapper3(chan_t)};
    wrapped();
}
