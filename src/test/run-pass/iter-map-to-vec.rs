fn inc(+x: uint) -> uint { x + 1 }

fn main() {
    assert [1, 3].map_to_vec(inc) == ~[2, 4];
    assert [1, 2, 3].map_to_vec(inc) == ~[2, 3, 4];
    assert None.map_to_vec(inc) == ~[];
    assert Some(1).map_to_vec(inc) == ~[2];
    assert Some(2).map_to_vec(inc) == ~[3];
}
