fn main() {
    let x = ~1;
    let y = ptr::addr_of(*x) as uint;

    let lam_copy = fn@[copy x]() -> uint { ptr::addr_of(*x) as uint };
    let lam_move = fn@[move x]() -> uint { ptr::addr_of(*x) as uint };
    assert lam_copy() != y;
    assert lam_move() == y;

    let x = ~2;
    let y = ptr::addr_of(*x) as uint;
    let snd_copy = fn~[copy x]() -> uint { ptr::addr_of(*x) as uint };
    let snd_move = fn~[move x]() -> uint { ptr::addr_of(*x) as uint };
    assert snd_copy() != y;
    assert snd_move() == y;
}
