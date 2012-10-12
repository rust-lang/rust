extern mod std;

trait methods {
    fn to_bytes() -> ~[u8];
}

impl (): methods {
    fn to_bytes() -> ~[u8] {
        vec::from_elem(0, 0)
    }
}

// the position of this function is significant! - if it comes before methods
// then it works, if it comes after it then it doesnt!
fn to_bools(bitv: {storage: ~[u64]}) -> ~[bool] {
    vec::from_fn(8, |i| {
        let w = i / 64;
        let b = i % 64;
        let x = 1u64 & (bitv.storage[w] >> b);
        x == 1u64
    })
}

fn main() {
    let bools = ~[false, false, true, false, false, true, true, false];
    let bools2 = to_bools({storage: ~[0b01100100]});

    for uint::range(0, 8) |i| {
        io::println(fmt!("%u => %u vs %u", i, bools[i] as uint, bools2[i] as uint));
    }

    assert bools == bools2;
}
