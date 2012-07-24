#[abi = "rust-intrinsic"]
extern mod rusti {
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

#[inline(always)]
fn atomic_xchng(&dst: int, src: int) -> int {
    rusti::atomic_xchng(dst, src)
}