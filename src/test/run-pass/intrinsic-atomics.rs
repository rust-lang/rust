#[abi = "rust-intrinsic"]
native mod rusti {
    fn atomic_xchng(&dst: int, src: int) -> int;
    fn atomic_xchng_acq(&dst: int, src: int) -> int;
    fn atomic_xchng_rel(&dst: int, src: int) -> int;
    
    fn atomic_add(&dst: int, src: int) -> int;
    fn atomic_add_acq(&dst: int, src: int) -> int;
    fn atomic_add_rel(&dst: int, src: int) -> int;
    
    fn atomic_sub(&dst: int, src: int) -> int;
    fn atomic_sub_acq(&dst: int, src: int) -> int;
    fn atomic_sub_rel(&dst: int, src: int) -> int;
}

fn main() {
    let mut x = 1;

    assert rusti::atomic_xchng(x, 0) == 1;
    assert x == 0;

    assert rusti::atomic_xchng_acq(x, 1) == 0;
    assert x == 1;

    assert rusti::atomic_xchng_rel(x, 0) == 1;
    assert x == 0;

    assert rusti::atomic_add(x, 1) == 0;
    assert rusti::atomic_add_acq(x, 1) == 1;
    assert rusti::atomic_add_rel(x, 1) == 2;
    assert x == 3;

    assert rusti::atomic_sub(x, 1) == 3;
    assert rusti::atomic_sub_acq(x, 1) == 2;
    assert rusti::atomic_sub_rel(x, 1) == 1;
    assert x == 0;
}
