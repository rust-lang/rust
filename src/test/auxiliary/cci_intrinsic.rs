#[legacy_exports];
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

#[inline(always)]
fn atomic_xchg(dst: &mut int, src: int) -> int {
    rusti::atomic_xchg(dst, src)
}
