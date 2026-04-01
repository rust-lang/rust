//! amdgpu intrinsics
//!
//! The reference is the [LLVM amdgpu guide] and the [LLVM implementation].
//! The order of intrinsics here follows the order in the [LLVM implementation].
//!
//! [LLVM amdgpu guide]: https://llvm.org/docs/AMDGPUUsage.html#llvm-ir-intrinsics
//! [LLVM implementation]: https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsAMDGPU.td

#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.amdgcn.workitem.id.x"]
    safe fn llvm_workitem_id_x() -> u32;
    #[link_name = "llvm.amdgcn.workitem.id.y"]
    safe fn llvm_workitem_id_y() -> u32;
    #[link_name = "llvm.amdgcn.workitem.id.z"]
    safe fn llvm_workitem_id_z() -> u32;

    #[link_name = "llvm.amdgcn.workgroup.id.x"]
    safe fn llvm_workgroup_id_x() -> u32;
    #[link_name = "llvm.amdgcn.workgroup.id.y"]
    safe fn llvm_workgroup_id_y() -> u32;
    #[link_name = "llvm.amdgcn.workgroup.id.z"]
    safe fn llvm_workgroup_id_z() -> u32;

    #[link_name = "llvm.amdgcn.groupstaticsize"]
    safe fn llvm_groupstaticsize() -> u32;
    #[link_name = "llvm.amdgcn.dispatch.id"]
    safe fn llvm_dispatch_id() -> u64;

    #[link_name = "llvm.amdgcn.wavefrontsize"]
    safe fn llvm_wavefrontsize() -> u32;

    #[link_name = "llvm.amdgcn.s.barrier"]
    safe fn llvm_s_barrier();
    #[link_name = "llvm.amdgcn.s.barrier.signal"]
    fn llvm_s_barrier_signal(barrier_type: i32);
    #[link_name = "llvm.amdgcn.s.barrier.signal.isfirst"]
    fn llvm_s_barrier_signal_isfirst(barrier_type: i32) -> bool;
    #[link_name = "llvm.amdgcn.s.barrier.wait"]
    fn llvm_s_barrier_wait(barrier_type: i16);
    #[link_name = "llvm.amdgcn.s.get.barrier.state"]
    fn llvm_s_get_barrier_state(barrier_type: i32) -> u32;
    #[link_name = "llvm.amdgcn.wave.barrier"]
    safe fn llvm_wave_barrier();
    #[link_name = "llvm.amdgcn.sched.barrier"]
    fn llvm_sched_barrier(mask: u32);
    #[link_name = "llvm.amdgcn.sched.group.barrier"]
    fn llvm_sched_group_barrier(mask: u32, size: u32, sync_id: u32);

    #[link_name = "llvm.amdgcn.s.sleep"]
    safe fn llvm_s_sleep(count: u32);

    #[link_name = "llvm.amdgcn.s.sethalt"]
    safe fn llvm_s_sethalt(value: u32) -> !;

    #[link_name = "llvm.amdgcn.s.getpc"]
    safe fn llvm_s_getpc() -> i64;

    #[link_name = "llvm.amdgcn.mbcnt.lo"]
    safe fn llvm_mbcnt_lo(value: u32, init: u32) -> u32;
    #[link_name = "llvm.amdgcn.mbcnt.hi"]
    safe fn llvm_mbcnt_hi(value: u32, init: u32) -> u32;

    #[link_name = "llvm.amdgcn.ballot"]
    safe fn llvm_ballot(b: bool) -> u64;

    #[link_name = "llvm.amdgcn.inverse.ballot"]
    safe fn llvm_inverse_ballot(value: u64) -> bool;

    #[link_name = "llvm.amdgcn.wave.reduce.umin"]
    safe fn llvm_wave_reduce_umin(value: u32, strategy: u32) -> u32;
    #[link_name = "llvm.amdgcn.wave.reduce.min"]
    safe fn llvm_wave_reduce_min(value: i32, strategy: u32) -> i32;
    #[link_name = "llvm.amdgcn.wave.reduce.umax"]
    safe fn llvm_wave_reduce_umax(value: u32, strategy: u32) -> u32;
    #[link_name = "llvm.amdgcn.wave.reduce.max"]
    safe fn llvm_wave_reduce_max(value: i32, strategy: u32) -> i32;
    #[link_name = "llvm.amdgcn.wave.reduce.add"]
    safe fn llvm_wave_reduce_add(value: u32, strategy: u32) -> u32;
    #[link_name = "llvm.amdgcn.wave.reduce.and"]
    safe fn llvm_wave_reduce_and(value: u32, strategy: u32) -> u32;
    #[link_name = "llvm.amdgcn.wave.reduce.or"]
    safe fn llvm_wave_reduce_or(value: u32, strategy: u32) -> u32;
    #[link_name = "llvm.amdgcn.wave.reduce.xor"]
    safe fn llvm_wave_reduce_xor(value: u32, strategy: u32) -> u32;

    // The following intrinsics can have multiple sizes

    #[link_name = "llvm.amdgcn.readfirstlane.i32"]
    safe fn llvm_readfirstlane_u32(value: u32) -> u32;
    #[link_name = "llvm.amdgcn.readfirstlane.i64"]
    safe fn llvm_readfirstlane_u64(value: u64) -> u64;
    #[link_name = "llvm.amdgcn.readlane.i32"]
    fn llvm_readlane_u32(value: u32, lane: u32) -> u32;
    #[link_name = "llvm.amdgcn.readlane.i64"]
    fn llvm_readlane_u64(value: u64, lane: u32) -> u64;
    #[link_name = "llvm.amdgcn.writelane.i32"]
    fn llvm_writelane_u32(value: u32, lane: u32, default: u32) -> u32;
    #[link_name = "llvm.amdgcn.writelane.i64"]
    fn llvm_writelane_u64(value: u64, lane: u32, default: u64) -> u64;

    #[link_name = "llvm.amdgcn.endpgm"]
    safe fn llvm_endpgm() -> !;

    #[link_name = "llvm.amdgcn.update.dpp.i32"]
    fn llvm_update_dpp(
        old: u32,
        src: u32,
        dpp_ctrl: u32,
        row_mask: u32,
        bank_mask: u32,
        bound_control: bool,
    ) -> u32;

    #[link_name = "llvm.amdgcn.s.memrealtime"]
    safe fn llvm_s_memrealtime() -> u64;

    #[link_name = "llvm.amdgcn.ds.permute"]
    fn llvm_ds_permute(lane: u32, value: u32) -> u32;
    #[link_name = "llvm.amdgcn.ds.bpermute"]
    fn llvm_ds_bpermute(lane: u32, value: u32) -> u32;
    #[link_name = "llvm.amdgcn.perm"]
    fn llvm_perm(src0: u32, src1: u32, selector: u32) -> u32;

    // gfx10
    #[link_name = "llvm.amdgcn.permlane16.i32"]
    fn llvm_permlane16_u32(
        old: u32,
        src0: u32,
        src1: u32,
        src2: u32,
        fi: bool,
        bound_control: bool,
    ) -> u32;

    // gfx10
    #[link_name = "llvm.amdgcn.permlanex16.i32"]
    fn llvm_permlanex16_u32(
        old: u32,
        src0: u32,
        src1: u32,
        src2: u32,
        fi: bool,
        bound_control: bool,
    ) -> u32;

    #[link_name = "llvm.amdgcn.s.get.waveid.in.workgroup"]
    safe fn llvm_s_get_waveid_in_workgroup() -> u32;

    // gfx11
    #[link_name = "llvm.amdgcn.permlane64.i32"]
    fn llvm_permlane64_u32(value: u32) -> u32;

    // gfx12
    #[link_name = "llvm.amdgcn.permlane16.var"]
    fn llvm_permlane16_var(old: u32, src0: u32, src1: u32, fi: bool, bound_control: bool) -> u32;

    // gfx12
    #[link_name = "llvm.amdgcn.permlanex16.var"]
    fn llvm_permlanex16_var(old: u32, src0: u32, src1: u32, fi: bool, bound_control: bool) -> u32;

    #[link_name = "llvm.amdgcn.wave.id"]
    safe fn llvm_wave_id() -> u32;

    // gfx950
    #[link_name = "llvm.amdgcn.permlane16.swap"]
    fn llvm_permlane16_swap(
        vdst_old: u32,
        vsrc_src0: u32,
        fi: bool,
        bound_control: bool,
    ) -> (u32, u32);

    // gfx950
    #[link_name = "llvm.amdgcn.permlane32.swap"]
    fn llvm_permlane32_swap(
        vdst_old: u32,
        vsrc_src0: u32,
        fi: bool,
        bound_control: bool,
    ) -> (u32, u32);
}

/// Returns the x coordinate of the workitem index within the workgroup.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn workitem_id_x() -> u32 {
    llvm_workitem_id_x()
}
/// Returns the y coordinate of the workitem index within the workgroup.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn workitem_id_y() -> u32 {
    llvm_workitem_id_y()
}
/// Returns the z coordinate of the workitem index within the workgroup.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn workitem_id_z() -> u32 {
    llvm_workitem_id_z()
}

/// Returns the x coordinate of the workgroup index within the dispatch.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn workgroup_id_x() -> u32 {
    llvm_workgroup_id_x()
}
/// Returns the y coordinate of the workgroup index within the dispatch.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn workgroup_id_y() -> u32 {
    llvm_workgroup_id_y()
}
/// Returns the z coordinate of the workgroup index within the dispatch.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn workgroup_id_z() -> u32 {
    llvm_workgroup_id_z()
}

/// Returns the size of statically allocated shared memory for this program in bytes.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn groupstaticsize() -> u32 {
    llvm_groupstaticsize()
}
/// Returns the id of the dispatch that is currently executed.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn dispatch_id() -> u64 {
    llvm_dispatch_id()
}

/// Returns the number of threads in a wavefront.
///
/// Is always a power of 2.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn wavefrontsize() -> u32 {
    llvm_wavefrontsize()
}

/// Synchronize all wavefronts in a workgroup.
///
/// Each wavefronts in a workgroup waits at the barrier until all wavefronts in the workgroup arrive at a barrier.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn s_barrier() {
    llvm_s_barrier()
}

/// Signal a specific barrier type.
///
/// Only for non-named barriers.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub unsafe fn s_barrier_signal<const BARRIER_TYPE: i32>() {
    unsafe { llvm_s_barrier_signal(BARRIER_TYPE) }
}

/// Signal a specific barrier type.
///
/// Only for non-named barriers.
/// Provides access to the s_barrier_signal_first instruction;
/// additionally ensures that the result value is valid even when
/// the intrinsic is used from a wavefront that is not running in a workgroup.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub unsafe fn s_barrier_signal_isfirst<const BARRIER_TYPE: i32>() -> bool {
    unsafe { llvm_s_barrier_signal_isfirst(BARRIER_TYPE) }
}

/// Wait for a specific barrier type.
///
/// Only for non-named barriers.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub unsafe fn s_barrier_wait<const BARRIER_TYPE: i16>() {
    unsafe { llvm_s_barrier_wait(BARRIER_TYPE) }
}

/// Get the state of a specific barrier type.
///
/// The `barrier_type` argument must be uniform, otherwise behavior is undefined.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub unsafe fn s_get_barrier_state<const BARRIER_TYPE: i32>() -> u32 {
    unsafe { llvm_s_get_barrier_state(BARRIER_TYPE) }
}

/// A barrier for only the threads within the current wavefront.
///
/// Does not result in an instruction but restricts the compiler.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn wave_barrier() {
    llvm_wave_barrier()
}

/// Prevent movement of some instruction types.
///
/// Controls the types of instructions that may be allowed to cross the intrinsic during instruction scheduling.
/// The parameter is a mask for the instruction types that can cross the intrinsic.
///
/// - 0x0000: No instructions may be scheduled across `sched_barrier`.
/// - 0x0001: All, non-memory, non-side-effect producing instructions may be scheduled across `sched_barrier`, i.e. allow ALU instructions to pass.
/// - 0x0002: VALU instructions may be scheduled across `sched_barrier`.
/// - 0x0004: SALU instructions may be scheduled across `sched_barrier`.
/// - 0x0008: MFMA/WMMA instructions may be scheduled across `sched_barrier`.
/// - 0x0010: All VMEM instructions may be scheduled across `sched_barrier`.
/// - 0x0020: VMEM read instructions may be scheduled across `sched_barrier`.
/// - 0x0040: VMEM write instructions may be scheduled across `sched_barrier`.
/// - 0x0080: All DS instructions may be scheduled across `sched_barrier`.
/// - 0x0100: All DS read instructions may be scheduled across `sched_barrier`.
/// - 0x0200: All DS write instructions may be scheduled across `sched_barrier`.
/// - 0x0400: All Transcendental (e.g. V_EXP) instructions may be scheduled across `sched_barrier`.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub unsafe fn sched_barrier<const MASK: u32>() {
    static_assert_uimm_bits!(MASK, 11);
    unsafe { llvm_sched_barrier(MASK) }
}

/// Creates schedule groups with specific properties to create custom scheduling pipelines.
///
/// The ordering between groups is enforced by the instruction scheduler.
/// The intrinsic applies to the code that precedes the intrinsic.
/// The intrinsic takes three values that control the behavior of the schedule groups.
///
/// - `mask`: Classify instruction groups using the [`sched_barrier`] mask values.
/// - `size`: The number of instructions that are in the group.
/// - `sync_id`: Order is enforced between groups with matching values.
///
/// The mask can include multiple instruction types. It is undefined behavior to set values beyond the range of valid masks.
///
/// Combining multiple `sched_group_barrier` intrinsics enables an ordering of specific instruction types during instruction scheduling.
/// For example, the following enforces a sequence of 1 VMEM read, followed by 1 VALU instruction, followed by 5 MFMA instructions.
///
/// ```rust
/// // 1 VMEM read
/// sched_group_barrier::<32, 1, 0>()
/// // 1 VALU
/// sched_group_barrier::<2, 1, 0>()
/// // 5 MFMA
/// sched_group_barrier::<8, 5, 0>()
/// ```
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub unsafe fn sched_group_barrier<const MASK: u32, const SIZE: u32, const SYNC_ID: u32>() {
    static_assert_uimm_bits!(MASK, 11);
    unsafe { llvm_sched_group_barrier(MASK, SIZE, SYNC_ID) }
}

/// Sleeps for approximately `COUNT * 64` cycles.
///
/// `COUNT` must be a constant.
/// Only the lower 7 bits of `COUNT` are used.
/// If `COUNT == 0x8000`, sleep forever until woken up, or killed.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn s_sleep<const COUNT: u32>() {
    llvm_s_sleep(COUNT)
}

/// Stop execution of the kernel.
///
/// This usually signals an error state.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn s_sethalt<const VALUE: u32>() -> ! {
    static_assert_uimm_bits!(VALUE, 3);
    llvm_s_sethalt(VALUE)
}

/// Returns the current process counter.
///
/// Provides access to the s_getpc_b64 instruction, but with the return value sign-extended
/// from the width of the underlying PC hardware register even on processors where the
/// s_getpc_b64 instruction returns a zero-extended value.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn s_getpc() -> i64 {
    llvm_s_getpc()
}

/// Masked bit count, low 32 lanes.
///
/// Computes the number of bits set in `value`, masked with a thread mask
/// which contains 1 for all active threads less than the current thread within a wavefront.
/// `init` is added to the result.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn mbcnt_lo(value: u32, init: u32) -> u32 {
    llvm_mbcnt_lo(value, init)
}
/// Masked bit count, high 32 lanes.
///
/// Computes the number of bits set in `value`, masked with a thread mask
/// which contains 1 for all active threads less than the current thread within a wavefront.
/// `init` is added to the result.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn mbcnt_hi(value: u32, init: u32) -> u32 {
    llvm_mbcnt_hi(value, init)
}

/// Returns a bitfield (`u32` or `u64`) containing the result of its i1 argument
/// in all active lanes, and zero in all inactive lanes.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn ballot(b: bool) -> u64 {
    llvm_ballot(b)
}

/// Indexes into the `value` with the current lane id and returns for each lane
/// if the corresponding bit is set.
///
/// While [`ballot`] converts a `bool` to a mask, `inverse_ballot` converts a mask back to a `bool`.
/// This means `inverse_ballot(ballot(b)) == b`.
/// The inverse of `ballot(inverse_ballot(value)) ~= value` is not always true as inactive lanes are set to zero by `ballot`.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn inverse_ballot(value: u64) -> bool {
    llvm_inverse_ballot(value)
}

/// Performs an arithmetic min reduction on the unsigned values provided by each lane in the wavefront.
///
/// The `STRATEGY` argument is a hint for the reduction strategy.
/// - 0: Target default preference
/// - 1: Iterative strategy
/// - 2: DPP
///
/// If target does not support the DPP operations (e.g. gfx6/7), reduction will be performed using default iterative strategy.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn wave_reduce_umin<const STRATEGY: u32>(value: u32) -> u32 {
    static_assert!(STRATEGY <= 2);
    llvm_wave_reduce_umin(value, STRATEGY)
}
/// Performs an arithmetic min reduction on the signed values provided by each lane in the wavefront.
///
/// The `STRATEGY` argument is a hint for the reduction strategy.
/// - 0: Target default preference
/// - 1: Iterative strategy
/// - 2: DPP
///
/// If target does not support the DPP operations (e.g. gfx6/7), reduction will be performed using default iterative strategy.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn wave_reduce_min<const STRATEGY: u32>(value: i32) -> i32 {
    static_assert!(STRATEGY <= 2);
    llvm_wave_reduce_min(value, STRATEGY)
}

/// Performs an arithmetic max reduction on the unsigned values provided by each lane in the wavefront.
///
/// The `STRATEGY` argument is a hint for the reduction strategy.
/// - 0: Target default preference
/// - 1: Iterative strategy
/// - 2: DPP
///
/// If target does not support the DPP operations (e.g. gfx6/7), reduction will be performed using default iterative strategy.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn wave_reduce_umax<const STRATEGY: u32>(value: u32) -> u32 {
    static_assert!(STRATEGY <= 2);
    llvm_wave_reduce_umax(value, STRATEGY)
}
/// Performs an arithmetic max reduction on the signed values provided by each lane in the wavefront.
///
/// The `STRATEGY` argument is a hint for the reduction strategy.
/// - 0: Target default preference
/// - 1: Iterative strategy
/// - 2: DPP
///
/// If target does not support the DPP operations (e.g. gfx6/7), reduction will be performed using default iterative strategy.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn wave_reduce_max<const STRATEGY: u32>(value: i32) -> i32 {
    static_assert!(STRATEGY <= 2);
    llvm_wave_reduce_max(value, STRATEGY)
}

/// Performs an arithmetic add reduction on the values provided by each lane in the wavefront.
///
/// The `STRATEGY` argument is a hint for the reduction strategy.
/// - 0: Target default preference
/// - 1: Iterative strategy
/// - 2: DPP
///
/// If target does not support the DPP operations (e.g. gfx6/7), reduction will be performed using default iterative strategy.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn wave_reduce_add<const STRATEGY: u32>(value: u32) -> u32 {
    static_assert!(STRATEGY <= 2);
    llvm_wave_reduce_add(value, STRATEGY)
}

/// Performs a logical and reduction on the unsigned values provided by each lane in the wavefront.
///
/// The `STRATEGY` argument is a hint for the reduction strategy.
/// - 0: Target default preference
/// - 1: Iterative strategy
/// - 2: DPP
///
/// If target does not support the DPP operations (e.g. gfx6/7), reduction will be performed using default iterative strategy.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn wave_reduce_and<const STRATEGY: u32>(value: u32) -> u32 {
    static_assert!(STRATEGY <= 2);
    llvm_wave_reduce_and(value, STRATEGY)
}
/// Performs a logical or reduction on the unsigned values provided by each lane in the wavefront.
///
/// The `STRATEGY` argument is a hint for the reduction strategy.
/// - 0: Target default preference
/// - 1: Iterative strategy
/// - 2: DPP
///
/// If target does not support the DPP operations (e.g. gfx6/7), reduction will be performed using default iterative strategy.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn wave_reduce_or<const STRATEGY: u32>(value: u32) -> u32 {
    static_assert!(STRATEGY <= 2);
    llvm_wave_reduce_or(value, STRATEGY)
}
/// Performs a logical xor reduction on the unsigned values provided by each lane in the wavefront.
///
/// The `STRATEGY` argument is a hint for the reduction strategy.
/// - 0: Target default preference
/// - 1: Iterative strategy
/// - 2: DPP
///
/// If target does not support the DPP operations (e.g. gfx6/7), reduction will be performed using default iterative strategy.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn wave_reduce_xor<const STRATEGY: u32>(value: u32) -> u32 {
    static_assert!(STRATEGY <= 2);
    llvm_wave_reduce_xor(value, STRATEGY)
}

// The following intrinsics can have multiple sizes

/// Get `value` from the first active lane in the wavefront.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn readfirstlane_u32(value: u32) -> u32 {
    llvm_readfirstlane_u32(value)
}
/// Get `value` from the first active lane in the wavefront.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn readfirstlane_u64(value: u64) -> u64 {
    llvm_readfirstlane_u64(value)
}
/// Get `value` from the lane at index `lane` in the wavefront.
///
/// The lane argument must be uniform across the currently active threads
/// of the current wavefront. Otherwise, the result is undefined.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub unsafe fn readlane_u32(value: u32, lane: u32) -> u32 {
    unsafe { llvm_readlane_u32(value, lane) }
}
/// Get `value` from the lane at index `lane` in the wavefront.
///
/// The lane argument must be uniform across the currently active threads
/// of the current wavefront. Otherwise, the result is undefined.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub unsafe fn readlane_u64(value: u64, lane: u32) -> u64 {
    unsafe { llvm_readlane_u64(value, lane) }
}
/// Return `value` for the lane at index `lane` in the wavefront.
/// Return `default` for all other lanes.
///
/// The value to write and lane select arguments must be uniform across the
/// currently active threads of the current wavefront. Otherwise, the result is
/// undefined.
///
/// `value` is the value returned by `lane`.
/// `default` is the value returned by all lanes other than `lane`.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub unsafe fn writelane_u32(value: u32, lane: u32, default: u32) -> u32 {
    unsafe { llvm_writelane_u32(value, lane, default) }
}
/// Return `value` for the lane at index `lane` in the wavefront.
/// Return `default` for all other lanes.
///
/// The value to write and lane select arguments must be uniform across the
/// currently active threads of the current wavefront. Otherwise, the result is
/// undefined.
///
/// `value` is the value returned by `lane`.
/// `default` is the value returned by all lanes other than `lane`.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub unsafe fn writelane_u64(value: u64, lane: u32, default: u64) -> u64 {
    unsafe { llvm_writelane_u64(value, lane, default) }
}

/// Stop execution of the wavefront.
///
/// This usually signals the end of a successful execution.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn endpgm() -> ! {
    llvm_endpgm()
}

/// The `update_dpp` intrinsic represents the `update.dpp` operation in AMDGPU.
/// It takes an old value, a source operand, a DPP control operand, a row mask, a bank mask, and a bound control.
/// This operation is equivalent to a sequence of `v_mov_b32` operations.
///
/// `llvm.amdgcn.update.dpp.i32 <old> <src> <dpp_ctrl> <row_mask> <bank_mask> <bound_ctrl>`
/// Should be equivalent to:
/// ```asm
/// v_mov_b32 <dest> <old>
/// v_mov_b32 <dest> <src> <dpp_ctrl> <row_mask> <bank_mask> <bound_ctrl>
/// ```
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub unsafe fn update_dpp<
    const DPP_CTRL: u32,
    const ROW_MASK: u32,
    const BANK_MASK: u32,
    const BOUND_CONTROL: bool,
>(
    old: u32,
    src: u32,
) -> u32 {
    unsafe { llvm_update_dpp(old, src, DPP_CTRL, ROW_MASK, BANK_MASK, BOUND_CONTROL) }
}

/// Measures time based on a fixed frequency.
///
/// Provides a real-time clock counter that runs at constant speed (typically 100 MHz) independent of ALU clock speeds.
/// The clock is consistent across the chip, so can be used for measuring between different wavefronts.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn s_memrealtime() -> u64 {
    llvm_s_memrealtime()
}

/// Scatter data across all lanes in a wavefront.
///
/// Writes `value` to the lane `lane`.
///
/// Reading from inactive lanes returns `0`.
/// In case multiple values get written to the same `lane`, the value from the source lane with the higher index is taken.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub unsafe fn ds_permute(lane: u32, value: u32) -> u32 {
    unsafe { llvm_ds_permute(lane, value) }
}
/// Gather data across all lanes in a wavefront.
///
/// Returns the `value` given to `ds_permute` by lane `lane`.
///
/// Reading from inactive lanes returns `0`.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub unsafe fn ds_bpermute(lane: u32, value: u32) -> u32 {
    unsafe { llvm_ds_bpermute(lane, value) }
}
/// Permute a 64-bit value.
///
/// `selector` selects between different patterns in which the 64-bit values represented by `src0` and `src1` are permuted.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub unsafe fn perm(src0: u32, src1: u32, selector: u32) -> u32 {
    unsafe { llvm_perm(src0, src1, selector) }
}

// gfx10
/// Performs arbitrary gather-style operation within a row (16 contiguous lanes) of the second input operand.
///
/// The third and fourth inputs must be uniform across the current wavefront.
/// These are combined into a single 64-bit value representing lane selects used to swizzle within each row.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub unsafe fn permlane16_u32<const FI: bool, const BOUND_CONTROL: bool>(
    old: u32,
    src0: u32,
    src1: u32,
    src2: u32,
) -> u32 {
    unsafe { llvm_permlane16_u32(old, src0, src1, src2, FI, BOUND_CONTROL) }
}

// gfx10
/// Performs arbitrary gather-style operation across two rows (16 contiguous lanes) of the second input operand.
///
/// The third and fourth inputs must be uniform across the current wavefront.
/// These are combined into a single 64-bit value representing lane selects used to swizzle within each row.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub unsafe fn permlanex16_u32<const FI: bool, const BOUND_CONTROL: bool>(
    old: u32,
    src0: u32,
    src1: u32,
    src2: u32,
) -> u32 {
    unsafe { llvm_permlanex16_u32(old, src0, src1, src2, FI, BOUND_CONTROL) }
}

/// Get the index of the current wavefront in the workgroup.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn s_get_waveid_in_workgroup() -> u32 {
    llvm_s_get_waveid_in_workgroup()
}

// gfx11
/// Swap `value` between upper and lower 32 lanes in a wavefront.
///
/// Does nothing for wave32.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub unsafe fn permlane64_u32(value: u32) -> u32 {
    unsafe { llvm_permlane64_u32(value) }
}

// gfx12
/// Performs arbitrary gather-style operation within a row (16 contiguous lanes) of the second input operand.
///
/// In contrast to [`permlane16_u32`], allows each lane to specify its own gather lane.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub unsafe fn permlane16_var<const FI: bool, const BOUND_CONTROL: bool>(
    old: u32,
    src0: u32,
    src1: u32,
) -> u32 {
    unsafe { llvm_permlane16_var(old, src0, src1, FI, BOUND_CONTROL) }
}

// gfx12
/// Performs arbitrary gather-style operation across two rows (16 contiguous lanes) of the second input operand.
///
/// In contrast to [`permlanex16_u32`], allows each lane to specify its own gather lane.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub unsafe fn permlanex16_var<const FI: bool, const BOUND_CONTROL: bool>(
    old: u32,
    src0: u32,
    src1: u32,
) -> u32 {
    unsafe { llvm_permlanex16_var(old, src0, src1, FI, BOUND_CONTROL) }
}

/// Get the index of the current wavefront in the workgroup.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub fn wave_id() -> u32 {
    llvm_wave_id()
}

// gfx950
/// Provide direct access to `v_permlane16_swap_b32` instruction on supported targets.
///
/// Swaps the values across lanes of first 2 operands.
/// Odd rows of the first operand are swapped with even rows of the second operand (one row is 16 lanes).
/// Returns a pair for the swapped registers.
/// The first element of the return corresponds to the swapped element of the first argument.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub unsafe fn permlane16_swap<const FI: bool, const BOUND_CONTROL: bool>(
    vdst_old: u32,
    vsrc_src0: u32,
) -> (u32, u32) {
    unsafe { llvm_permlane16_swap(vdst_old, vsrc_src0, FI, BOUND_CONTROL) }
}

// gfx950
/// Provide direct access to `v_permlane32_swap_b32` instruction on supported targets.
///
/// Swaps the values across lanes of first 2 operands.
/// Rows 2 and 3 of the first operand are swapped with rows 0 and 1 of the second operand (one row is 16 lanes).
/// Returns a pair for the swapped registers.
/// The first element of the return corresponds to the swapped element of the first argument.
#[inline]
#[unstable(feature = "stdarch_amdgpu", issue = "149988")]
pub unsafe fn permlane32_swap<const FI: bool, const BOUND_CONTROL: bool>(
    vdst_old: u32,
    vsrc_src0: u32,
) -> (u32, u32) {
    unsafe { llvm_permlane32_swap(vdst_old, vsrc_src0, FI, BOUND_CONTROL) }
}

// Functions to generate code, used to check that the intrinsics build.
// Marked as no_mangle, so the compiler does not remove the functions.
// To test, uncomment the `#[cfg(test)]` line below and run
// NORUN=1 NOSTD=1 TARGET=amdgcn-amd-amdhsa CARGO_UNSTABLE_BUILD_STD=core ci/run.sh
//
// Note that depending on the target-cpu set in run.sh, some of these intrinsics are not available
// and compilation fails with `Cannot select: intrinsic %llvm.amdgcn...`.
// Uncomment these intrinsics to check.
#[cfg(test)]
mod tests {
    use super::*;

    #[unsafe(no_mangle)]
    fn test_workitem_id_x() -> u32 {
        workitem_id_x()
    }
    #[unsafe(no_mangle)]
    fn test_workitem_id_y() -> u32 {
        workitem_id_y()
    }
    #[unsafe(no_mangle)]
    fn test_workitem_id_z() -> u32 {
        workitem_id_z()
    }

    #[unsafe(no_mangle)]
    fn test_workgroup_id_x() -> u32 {
        workgroup_id_x()
    }
    #[unsafe(no_mangle)]
    fn test_workgroup_id_y() -> u32 {
        workgroup_id_y()
    }
    #[unsafe(no_mangle)]
    fn test_workgroup_id_z() -> u32 {
        workgroup_id_z()
    }

    #[unsafe(no_mangle)]
    fn test_groupstaticsize() -> u32 {
        groupstaticsize()
    }
    #[unsafe(no_mangle)]
    fn test_dispatch_id() -> u64 {
        dispatch_id()
    }

    #[unsafe(no_mangle)]
    fn test_wavefrontsize() -> u32 {
        wavefrontsize()
    }

    #[unsafe(no_mangle)]
    fn test_s_barrier() {
        s_barrier()
    }

    #[unsafe(no_mangle)]
    fn test_s_barrier_signal() {
        unsafe { s_barrier_signal::<-1>() }
    }

    #[unsafe(no_mangle)]
    fn test_s_barrier_signal_isfirst() -> bool {
        unsafe { s_barrier_signal_isfirst::<-1>() }
    }

    #[unsafe(no_mangle)]
    fn test_s_barrier_wait() {
        unsafe { s_barrier_wait::<-1>() }
    }

    #[unsafe(no_mangle)]
    fn test_s_get_barrier_state() -> u32 {
        unsafe { s_get_barrier_state::<-1>() }
    }

    #[unsafe(no_mangle)]
    fn test_wave_barrier() {
        wave_barrier()
    }

    #[unsafe(no_mangle)]
    fn test_sched_barrier() {
        unsafe { sched_barrier::<1>() }
    }

    #[unsafe(no_mangle)]
    fn test_sched_group_barrier() {
        unsafe { sched_group_barrier::<1, 1, 0>() }
    }

    #[unsafe(no_mangle)]
    fn test_s_sleep() {
        s_sleep::<1>()
    }

    #[unsafe(no_mangle)]
    fn test_s_sethalt() -> ! {
        s_sethalt::<1>()
    }

    #[unsafe(no_mangle)]
    fn test_s_getpc() -> i64 {
        s_getpc()
    }

    #[unsafe(no_mangle)]
    fn test_mbcnt_lo(value: u32, init: u32) -> u32 {
        mbcnt_lo(value, init)
    }
    #[unsafe(no_mangle)]
    fn test_mbcnt_hi(value: u32, init: u32) -> u32 {
        mbcnt_hi(value, init)
    }

    #[unsafe(no_mangle)]
    fn test_ballot(b: bool) -> u64 {
        ballot(b)
    }

    #[unsafe(no_mangle)]
    fn test_inverse_ballot(value: u64) -> bool {
        inverse_ballot(value)
    }

    #[unsafe(no_mangle)]
    fn test_wave_reduce_umin(value: u32) -> u32 {
        wave_reduce_umin::<0>(value)
    }
    #[unsafe(no_mangle)]
    fn test_wave_reduce_min(value: i32) -> i32 {
        wave_reduce_min::<0>(value)
    }

    #[unsafe(no_mangle)]
    fn test_wave_reduce_umax(value: u32) -> u32 {
        wave_reduce_umax::<0>(value)
    }
    #[unsafe(no_mangle)]
    fn test_wave_reduce_max(value: i32) -> i32 {
        wave_reduce_max::<0>(value)
    }

    #[unsafe(no_mangle)]
    fn test_wave_reduce_add(value: u32) -> u32 {
        wave_reduce_add::<0>(value)
    }

    #[unsafe(no_mangle)]
    fn test_wave_reduce_and(value: u32) -> u32 {
        wave_reduce_and::<0>(value)
    }
    #[unsafe(no_mangle)]
    fn test_wave_reduce_or(value: u32) -> u32 {
        wave_reduce_or::<0>(value)
    }
    #[unsafe(no_mangle)]
    fn test_wave_reduce_xor(value: u32) -> u32 {
        wave_reduce_xor::<0>(value)
    }

    #[unsafe(no_mangle)]
    fn test_readfirstlane_u32(value: u32) -> u32 {
        readfirstlane_u32(value)
    }
    #[unsafe(no_mangle)]
    fn test_readfirstlane_u64(value: u64) -> u64 {
        readfirstlane_u64(value)
    }
    #[unsafe(no_mangle)]
    fn test_readlane_u32(value: u32, lane: u32) -> u32 {
        unsafe { readlane_u32(value, lane) }
    }
    #[unsafe(no_mangle)]
    fn test_readlane_u64(value: u64, lane: u32) -> u64 {
        unsafe { readlane_u64(value, lane) }
    }
    #[unsafe(no_mangle)]
    fn test_writelane_u32(value: u32, lane: u32, default: u32) -> u32 {
        unsafe { writelane_u32(value, lane, default) }
    }
    #[unsafe(no_mangle)]
    fn test_writelane_u64(value: u64, lane: u32, default: u64) -> u64 {
        unsafe { writelane_u64(value, lane, default) }
    }

    #[unsafe(no_mangle)]
    fn test_endpgm() -> ! {
        endpgm()
    }

    #[unsafe(no_mangle)]
    fn test_update_dpp(old: u32, src: u32) -> u32 {
        unsafe { update_dpp::<0, 0, 0, true>(old, src) }
    }

    #[unsafe(no_mangle)]
    fn test_s_memrealtime() -> u64 {
        s_memrealtime()
    }

    #[unsafe(no_mangle)]
    fn test_ds_permute(lane: u32, value: u32) -> u32 {
        unsafe { ds_permute(lane, value) }
    }
    #[unsafe(no_mangle)]
    fn test_ds_bpermute(lane: u32, value: u32) -> u32 {
        unsafe { ds_bpermute(lane, value) }
    }
    #[unsafe(no_mangle)]
    fn test_perm(src0: u32, src1: u32, selector: u32) -> u32 {
        unsafe { perm(src0, src1, selector) }
    }

    #[unsafe(no_mangle)]
    fn test_permlane16_u32(old: u32, src0: u32, src1: u32, src2: u32) -> u32 {
        unsafe { permlane16_u32::<false, true>(old, src0, src1, src2) }
    }

    #[unsafe(no_mangle)]
    fn test_permlanex16_u32(old: u32, src0: u32, src1: u32, src2: u32) -> u32 {
        unsafe { permlanex16_u32::<false, true>(old, src0, src1, src2) }
    }

    #[unsafe(no_mangle)]
    fn test_s_get_waveid_in_workgroup() -> u32 {
        s_get_waveid_in_workgroup()
    }

    #[unsafe(no_mangle)]
    fn test_permlane64_u32(value: u32) -> u32 {
        unsafe { permlane64_u32(value) }
    }

    #[unsafe(no_mangle)]
    fn test_permlane16_var(old: u32, src0: u32, src1: u32) -> u32 {
        unsafe { permlane16_var::<false, true>(old, src0, src1) }
    }

    #[unsafe(no_mangle)]
    fn test_permlanex16_var(old: u32, src0: u32, src1: u32) -> u32 {
        unsafe { permlanex16_var::<false, true>(old, src0, src1) }
    }

    #[unsafe(no_mangle)]
    fn test_wave_id() -> u32 {
        wave_id()
    }

    #[unsafe(no_mangle)]
    fn test_permlane16_swap(vdst_old: u32, vsrc_src0: u32) -> (u32, u32) {
        unsafe { permlane16_swap::<false, true>(vdst_old, vsrc_src0) }
    }

    #[unsafe(no_mangle)]
    fn test_permlane32_swap(vdst_old: u32, vsrc_src0: u32) -> (u32, u32) {
        unsafe { permlane32_swap::<false, true>(vdst_old, vsrc_src0) }
    }
}
