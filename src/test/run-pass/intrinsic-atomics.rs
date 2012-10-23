#[abi = "rust-intrinsic"]
extern mod rusti {
    #[legacy_exports];
    fn atomic_cxchg(dst: &mut int, old: int, src: int) -> int;
    fn atomic_cxchg_acq(dst: &mut int, old: int, src: int) -> int;
    fn atomic_cxchg_rel(dst: &mut int, old: int, src: int) -> int;

    fn atomic_xchg(dst: &mut int, src: int) -> int;
    fn atomic_xchg_acq(dst: &mut int, src: int) -> int;
    fn atomic_xchg_rel(dst: &mut int, src: int) -> int;
    
    fn atomic_xadd(dst: &mut int, src: int) -> int;
    fn atomic_xadd_acq(dst: &mut int, src: int) -> int;
    fn atomic_xadd_rel(dst: &mut int, src: int) -> int;
    
    fn atomic_xsub(dst: &mut int, src: int) -> int;
    fn atomic_xsub_acq(dst: &mut int, src: int) -> int;
    fn atomic_xsub_rel(dst: &mut int, src: int) -> int;
}

fn main() {
    let x = ~mut 1;

    assert rusti::atomic_cxchg(x, 1, 2) == 1;
    assert *x == 2;

    assert rusti::atomic_cxchg_acq(x, 1, 3) == 2;
    assert *x == 2;

    assert rusti::atomic_cxchg_rel(x, 2, 1) == 2;
    assert *x == 1;

    assert rusti::atomic_xchg(x, 0) == 1;
    assert *x == 0;

    assert rusti::atomic_xchg_acq(x, 1) == 0;
    assert *x == 1;

    assert rusti::atomic_xchg_rel(x, 0) == 1;
    assert *x == 0;

    assert rusti::atomic_xadd(x, 1) == 0;
    assert rusti::atomic_xadd_acq(x, 1) == 1;
    assert rusti::atomic_xadd_rel(x, 1) == 2;
    assert *x == 3;

    assert rusti::atomic_xsub(x, 1) == 3;
    assert rusti::atomic_xsub_acq(x, 1) == 2;
    assert rusti::atomic_xsub_rel(x, 1) == 1;
    assert *x == 0;
}
