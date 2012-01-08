fn main() {
    let x = ~1;
    let y = ptr::addr_of(*x) as uint;

    let lam_copy = lambda[copy x]() -> uint { ptr::addr_of(*x) as uint };
    let lam_move = lambda[move x]() -> uint { ptr::addr_of(*x) as uint };
    assert lam_copy() != y;
    assert lam_move() == y;

    let x = ~2;
    let y = ptr::addr_of(*x) as uint;
    let snd_copy = sendfn[copy x]() -> uint { ptr::addr_of(*x) as uint };
    let snd_move = sendfn[move x]() -> uint { ptr::addr_of(*x) as uint };
    assert snd_copy() != y;
    assert snd_move() == y;
}
