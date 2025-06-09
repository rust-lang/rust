//! This module is responsible for managing the absolute addresses that allocations are located at,
//! and for casting between pointers and integers based on those addresses.

mod reuse_pool;

use std::cell::RefCell;
use std::cmp::max;

use rand::Rng;
use rustc_abi::{Align, Size};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};

use self::reuse_pool::ReusePool;
use crate::concurrency::VClock;
use crate::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ProvenanceMode {
    /// We support `expose_provenance`/`with_exposed_provenance` via "wildcard" provenance.
    /// However, we warn on `with_exposed_provenance` to alert the user of the precision loss.
    Default,
    /// Like `Default`, but without the warning.
    Permissive,
    /// We error on `with_exposed_provenance`, ensuring no precision loss.
    Strict,
}

pub type GlobalState = RefCell<GlobalStateInner>;

#[derive(Debug)]
pub struct GlobalStateInner {
    /// This is used as a map between the address of each allocation and its `AllocId`. It is always
    /// sorted by address. We cannot use a `HashMap` since we can be given an address that is offset
    /// from the base address, and we need to find the `AllocId` it belongs to. This is not the
    /// *full* inverse of `base_addr`; dead allocations have been removed.
    int_to_ptr_map: Vec<(u64, AllocId)>,
    /// The base address for each allocation.  We cannot put that into
    /// `AllocExtra` because function pointers also have a base address, and
    /// they do not have an `AllocExtra`.
    /// This is the inverse of `int_to_ptr_map`.
    base_addr: FxHashMap<AllocId, u64>,
    /// Temporarily store prepared memory space for global allocations the first time their memory
    /// address is required. This is used to ensure that the memory is allocated before Miri assigns
    /// it an internal address, which is important for matching the internal address to the machine
    /// address so FFI can read from pointers.
    prepared_alloc_bytes: FxHashMap<AllocId, MiriAllocBytes>,
    /// A pool of addresses we can reuse for future allocations.
    reuse: ReusePool,
    /// Whether an allocation has been exposed or not. This cannot be put
    /// into `AllocExtra` for the same reason as `base_addr`.
    exposed: FxHashSet<AllocId>,
    /// This is used as a memory address when a new pointer is casted to an integer. It
    /// is always larger than any address that was previously made part of a block.
    next_base_addr: u64,
    /// The provenance to use for int2ptr casts
    provenance_mode: ProvenanceMode,
}

impl VisitProvenance for GlobalStateInner {
    fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {
        let GlobalStateInner {
            int_to_ptr_map: _,
            base_addr: _,
            prepared_alloc_bytes: _,
            reuse: _,
            exposed: _,
            next_base_addr: _,
            provenance_mode: _,
        } = self;
        // Though base_addr, int_to_ptr_map, and exposed contain AllocIds, we do not want to visit them.
        // int_to_ptr_map and exposed must contain only live allocations, and those
        // are never garbage collected.
        // base_addr is only relevant if we have a pointer to an AllocId and need to look up its
        // base address; so if an AllocId is not reachable from somewhere else we can remove it
        // here.
    }
}

impl GlobalStateInner {
    pub fn new(config: &MiriConfig, stack_addr: u64) -> Self {
        GlobalStateInner {
            int_to_ptr_map: Vec::default(),
            base_addr: FxHashMap::default(),
            prepared_alloc_bytes: FxHashMap::default(),
            reuse: ReusePool::new(config),
            exposed: FxHashSet::default(),
            next_base_addr: stack_addr,
            provenance_mode: config.provenance_mode,
        }
    }

    pub fn remove_unreachable_allocs(&mut self, allocs: &LiveAllocs<'_, '_>) {
        // `exposed` and `int_to_ptr_map` are cleared immediately when an allocation
        // is freed, so `base_addr` is the only one we have to clean up based on the GC.
        self.base_addr.retain(|id, _| allocs.is_live(*id));
    }
}

/// Shifts `addr` to make it aligned with `align` by rounding `addr` to the smallest multiple
/// of `align` that is larger or equal to `addr`
fn align_addr(addr: u64, align: u64) -> u64 {
    match addr % align {
        0 => addr,
        rem => addr.strict_add(align) - rem,
    }
}

impl<'tcx> EvalContextExtPriv<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextExtPriv<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn addr_from_alloc_id_uncached(
        &self,
        global_state: &mut GlobalStateInner,
        alloc_id: AllocId,
        memory_kind: MemoryKind,
    ) -> InterpResult<'tcx, u64> {
        let this = self.eval_context_ref();
        let info = this.get_alloc_info(alloc_id);

        // Miri's address assignment leaks state across thread boundaries, which is incompatible
        // with GenMC execution. So we instead let GenMC assign addresses to allocations.
        if let Some(genmc_ctx) = this.machine.data_race.as_genmc_ref() {
            let addr = genmc_ctx.handle_alloc(&this.machine, info.size, info.align, memory_kind)?;
            return interp_ok(addr);
        }

        let mut rng = this.machine.rng.borrow_mut();
        // This is either called immediately after allocation (and then cached), or when
        // adjusting `tcx` pointers (which never get freed). So assert that we are looking
        // at a live allocation. This also ensures that we never re-assign an address to an
        // allocation that previously had an address, but then was freed and the address
        // information was removed.
        assert!(!matches!(info.kind, AllocKind::Dead));

        // This allocation does not have a base address yet, pick or reuse one.
        if this.machine.native_lib.is_some() {
            // In native lib mode, we use the "real" address of the bytes for this allocation.
            // This ensures the interpreted program and native code have the same view of memory.
            let params = this.machine.get_default_alloc_params();
            let base_ptr = match info.kind {
                AllocKind::LiveData => {
                    if memory_kind == MiriMemoryKind::Global.into() {
                        // For new global allocations, we always pre-allocate the memory to be able use the machine address directly.
                        let prepared_bytes = MiriAllocBytes::zeroed(info.size, info.align, params)
                            .unwrap_or_else(|| {
                                panic!("Miri ran out of memory: cannot create allocation of {size:?} bytes", size = info.size)
                            });
                        let ptr = prepared_bytes.as_ptr();
                        // Store prepared allocation to be picked up for use later.
                        global_state
                            .prepared_alloc_bytes
                            .try_insert(alloc_id, prepared_bytes)
                            .unwrap();
                        ptr
                    } else {
                        // Non-global allocations are already in memory at this point so
                        // we can just get a pointer to where their data is stored.
                        this.get_alloc_bytes_unchecked_raw(alloc_id)?
                    }
                }
                AllocKind::Function | AllocKind::VTable => {
                    // Allocate some dummy memory to get a unique address for this function/vtable.
                    let alloc_bytes = MiriAllocBytes::from_bytes(
                        &[0u8; 1],
                        Align::from_bytes(1).unwrap(),
                        params,
                    );
                    let ptr = alloc_bytes.as_ptr();
                    // Leak the underlying memory to ensure it remains unique.
                    std::mem::forget(alloc_bytes);
                    ptr
                }
                AllocKind::Dead => unreachable!(),
            };
            // We don't have to expose this pointer yet, we do that in `prepare_for_native_call`.
            return interp_ok(base_ptr.addr().to_u64());
        }
        // We are not in native lib mode, so we control the addresses ourselves.
        if let Some((reuse_addr, clock)) = global_state.reuse.take_addr(
            &mut *rng,
            info.size,
            info.align,
            memory_kind,
            this.active_thread(),
        ) {
            if let Some(clock) = clock {
                this.acquire_clock(&clock);
            }
            interp_ok(reuse_addr)
        } else {
            // We have to pick a fresh address.
            // Leave some space to the previous allocation, to give it some chance to be less aligned.
            // We ensure that `(global_state.next_base_addr + slack) % 16` is uniformly distributed.
            let slack = rng.random_range(0..16);
            // From next_base_addr + slack, round up to adjust for alignment.
            let base_addr = global_state
                .next_base_addr
                .checked_add(slack)
                .ok_or_else(|| err_exhaust!(AddressSpaceFull))?;
            let base_addr = align_addr(base_addr, info.align.bytes());

            // Remember next base address.  If this allocation is zero-sized, leave a gap of at
            // least 1 to avoid two allocations having the same base address. (The logic in
            // `alloc_id_from_addr` assumes unique addresses, and different function/vtable pointers
            // need to be distinguishable!)
            global_state.next_base_addr = base_addr
                .checked_add(max(info.size.bytes(), 1))
                .ok_or_else(|| err_exhaust!(AddressSpaceFull))?;
            // Even if `Size` didn't overflow, we might still have filled up the address space.
            if global_state.next_base_addr > this.target_usize_max() {
                throw_exhaust!(AddressSpaceFull);
            }
            // If we filled up more than half the address space, start aggressively reusing
            // addresses to avoid running out.
            if global_state.next_base_addr > u64::try_from(this.target_isize_max()).unwrap() {
                global_state.reuse.address_space_shortage();
            }

            interp_ok(base_addr)
        }
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    // Returns the `AllocId` that corresponds to the specified addr,
    // or `None` if the addr is out of bounds.
    // Setting `only_exposed_allocations` selects whether only exposed allocations are considered.
    fn alloc_id_from_addr(
        &self,
        addr: u64,
        size: i64,
        only_exposed_allocations: bool,
    ) -> Option<AllocId> {
        let this = self.eval_context_ref();
        let global_state = this.machine.alloc_addresses.borrow();
        assert!(global_state.provenance_mode != ProvenanceMode::Strict);

        // We always search the allocation to the right of this address. So if the size is strictly
        // negative, we have to search for `addr-1` instead.
        let addr = if size >= 0 { addr } else { addr.saturating_sub(1) };
        let pos = global_state.int_to_ptr_map.binary_search_by_key(&addr, |(addr, _)| *addr);

        // Determine the in-bounds provenance for this pointer.
        let alloc_id = match pos {
            Ok(pos) => Some(global_state.int_to_ptr_map[pos].1),
            Err(0) => None,
            Err(pos) => {
                // This is the largest of the addresses smaller than `int`,
                // i.e. the greatest lower bound (glb)
                let (glb, alloc_id) = global_state.int_to_ptr_map[pos - 1];
                // This never overflows because `addr >= glb`
                let offset = addr - glb;
                // We require this to be strict in-bounds of the allocation. This arm is only
                // entered for addresses that are not the base address, so even zero-sized
                // allocations will get recognized at their base address -- but all other
                // allocations will *not* be recognized at their "end" address.
                let size = this.get_alloc_info(alloc_id).size;
                if offset < size.bytes() { Some(alloc_id) } else { None }
            }
        }?;

        // We only use this provenance if it has been exposed, or if the caller requested also non-exposed allocations
        if !only_exposed_allocations || global_state.exposed.contains(&alloc_id) {
            // This must still be live, since we remove allocations from `int_to_ptr_map` when they get freed.
            debug_assert!(this.is_alloc_live(alloc_id));
            Some(alloc_id)
        } else {
            None
        }
    }

    /// Returns the base address of an allocation, or an error if no base address could be found
    ///
    /// # Panics
    /// If `memory_kind = None` and the `alloc_id` is not cached, meaning that the first call to this function per `alloc_id` must get the `memory_kind`.
    fn addr_from_alloc_id(
        &self,
        alloc_id: AllocId,
        memory_kind: Option<MemoryKind>,
    ) -> InterpResult<'tcx, u64> {
        let this = self.eval_context_ref();
        let mut global_state = this.machine.alloc_addresses.borrow_mut();
        let global_state = &mut *global_state;

        match global_state.base_addr.get(&alloc_id) {
            Some(&addr) => interp_ok(addr),
            None => {
                // First time we're looking for the absolute address of this allocation.
                let memory_kind =
                    memory_kind.expect("memory_kind is required since alloc_id is not cached");
                let base_addr =
                    this.addr_from_alloc_id_uncached(global_state, alloc_id, memory_kind)?;
                trace!("Assigning base address {:#x} to allocation {:?}", base_addr, alloc_id);

                // Store address in cache.
                global_state.base_addr.try_insert(alloc_id, base_addr).unwrap();

                // Also maintain the opposite mapping in `int_to_ptr_map`, ensuring we keep it sorted.
                // We have a fast-path for the common case that this address is bigger than all previous ones.
                let pos = if global_state
                    .int_to_ptr_map
                    .last()
                    .is_some_and(|(last_addr, _)| *last_addr < base_addr)
                {
                    global_state.int_to_ptr_map.len()
                } else {
                    global_state
                        .int_to_ptr_map
                        .binary_search_by_key(&base_addr, |(addr, _)| *addr)
                        .unwrap_err()
                };
                global_state.int_to_ptr_map.insert(pos, (base_addr, alloc_id));

                interp_ok(base_addr)
            }
        }
    }

    fn expose_provenance(&self, provenance: Provenance) -> InterpResult<'tcx> {
        let this = self.eval_context_ref();
        let mut global_state = this.machine.alloc_addresses.borrow_mut();

        let (alloc_id, tag) = match provenance {
            Provenance::Concrete { alloc_id, tag } => (alloc_id, tag),
            Provenance::Wildcard => {
                // No need to do anything for wildcard pointers as
                // their provenances have already been previously exposed.
                return interp_ok(());
            }
        };

        // In strict mode, we don't need this, so we can save some cycles by not tracking it.
        if global_state.provenance_mode == ProvenanceMode::Strict {
            return interp_ok(());
        }
        // Exposing a dead alloc is a no-op, because it's not possible to get a dead allocation
        // via int2ptr.
        if !this.is_alloc_live(alloc_id) {
            return interp_ok(());
        }
        trace!("Exposing allocation id {alloc_id:?}");
        global_state.exposed.insert(alloc_id);
        // Release the global state before we call `expose_tag`, which may call `get_alloc_info_extra`,
        // which may need access to the global state.
        drop(global_state);
        if this.machine.borrow_tracker.is_some() {
            this.expose_tag(alloc_id, tag)?;
        }
        interp_ok(())
    }

    fn ptr_from_addr_cast(&self, addr: u64) -> InterpResult<'tcx, Pointer> {
        trace!("Casting {:#x} to a pointer", addr);

        let this = self.eval_context_ref();
        let global_state = this.machine.alloc_addresses.borrow();

        // Potentially emit a warning.
        match global_state.provenance_mode {
            ProvenanceMode::Default => {
                // The first time this happens at a particular location, print a warning.
                let mut int2ptr_warned = this.machine.int2ptr_warned.borrow_mut();
                let first = int2ptr_warned.is_empty();
                if int2ptr_warned.insert(this.cur_span()) {
                    // Newly inserted, so first time we see this span.
                    this.emit_diagnostic(NonHaltingDiagnostic::Int2Ptr { details: first });
                }
            }
            ProvenanceMode::Strict => {
                throw_machine_stop!(TerminationInfo::Int2PtrWithStrictProvenance);
            }
            ProvenanceMode::Permissive => {}
        }

        // We do *not* look up the `AllocId` here! This is a `ptr as usize` cast, and it is
        // completely legal to do a cast and then `wrapping_offset` to another allocation and only
        // *then* do a memory access. So the allocation that the pointer happens to point to on a
        // cast is fairly irrelevant. Instead we generate this as a "wildcard" pointer, such that
        // *every time the pointer is used*, we do an `AllocId` lookup to find the (exposed)
        // allocation it might be referencing.
        interp_ok(Pointer::new(Some(Provenance::Wildcard), Size::from_bytes(addr)))
    }

    /// Convert a relative (tcx) pointer to a Miri pointer.
    fn adjust_alloc_root_pointer(
        &self,
        ptr: interpret::Pointer<CtfeProvenance>,
        tag: BorTag,
        kind: MemoryKind,
    ) -> InterpResult<'tcx, interpret::Pointer<Provenance>> {
        let this = self.eval_context_ref();

        let (prov, offset) = ptr.into_parts(); // offset is relative (AllocId provenance)
        let alloc_id = prov.alloc_id();

        // Get a pointer to the beginning of this allocation.
        let base_addr = this.addr_from_alloc_id(alloc_id, Some(kind))?;
        let base_ptr = interpret::Pointer::new(
            Provenance::Concrete { alloc_id, tag },
            Size::from_bytes(base_addr),
        );
        // Add offset with the right kind of pointer-overflowing arithmetic.
        interp_ok(base_ptr.wrapping_offset(offset, this))
    }

    // This returns some prepared `MiriAllocBytes`, either because `addr_from_alloc_id` reserved
    // memory space in the past, or by doing the pre-allocation right upon being called.
    fn get_global_alloc_bytes(
        &self,
        id: AllocId,
        bytes: &[u8],
        align: Align,
    ) -> InterpResult<'tcx, MiriAllocBytes> {
        let this = self.eval_context_ref();
        assert!(this.tcx.try_get_global_alloc(id).is_some());
        if this.machine.native_lib.is_some() {
            // In native lib mode, MiriAllocBytes for global allocations are handled via `prepared_alloc_bytes`.
            // This additional call ensures that some `MiriAllocBytes` are always prepared, just in case
            // this function gets called before the first time `addr_from_alloc_id` gets called.
            this.addr_from_alloc_id(id, Some(MiriMemoryKind::Global.into()))?;
            // The memory we need here will have already been allocated during an earlier call to
            // `addr_from_alloc_id` for this allocation. So don't create a new `MiriAllocBytes` here, instead
            // fetch the previously prepared bytes from `prepared_alloc_bytes`.
            let mut global_state = this.machine.alloc_addresses.borrow_mut();
            let mut prepared_alloc_bytes = global_state
                .prepared_alloc_bytes
                .remove(&id)
                .unwrap_or_else(|| panic!("alloc bytes for {id:?} have not been prepared"));
            // Sanity-check that the prepared allocation has the right size and alignment.
            assert!(prepared_alloc_bytes.as_ptr().is_aligned_to(align.bytes_usize()));
            assert_eq!(prepared_alloc_bytes.len(), bytes.len());
            // Copy allocation contents into prepared memory.
            prepared_alloc_bytes.copy_from_slice(bytes);
            interp_ok(prepared_alloc_bytes)
        } else {
            let params = this.machine.get_default_alloc_params();
            interp_ok(MiriAllocBytes::from_bytes(std::borrow::Cow::Borrowed(bytes), align, params))
        }
    }

    /// When a pointer is used for a memory access, this computes where in which allocation the
    /// access is going.
    fn ptr_get_alloc(
        &self,
        ptr: interpret::Pointer<Provenance>,
        size: i64,
    ) -> Option<(AllocId, Size)> {
        let this = self.eval_context_ref();

        let (tag, addr) = ptr.into_parts(); // addr is absolute (Tag provenance)

        let alloc_id = if let Provenance::Concrete { alloc_id, .. } = tag {
            alloc_id
        } else {
            // A wildcard pointer.
            let only_exposed_allocations = true;
            this.alloc_id_from_addr(addr.bytes(), size, only_exposed_allocations)?
        };

        // This cannot fail: since we already have a pointer with that provenance, adjust_alloc_root_pointer
        // must have been called in the past, so we can just look up the address in the map.
        let base_addr = *this.machine.alloc_addresses.borrow().base_addr.get(&alloc_id).unwrap();

        // Wrapping "addr - base_addr"
        let rel_offset = this.truncate_to_target_usize(addr.bytes().wrapping_sub(base_addr));
        Some((alloc_id, Size::from_bytes(rel_offset)))
    }

    /// Prepare all exposed memory for a native call.
    /// This overapproximates the modifications which external code might make to memory:
    /// We set all reachable allocations as initialized, mark all reachable provenances as exposed
    /// and overwrite them with `Provenance::WILDCARD`.
    fn prepare_exposed_for_native_call(&mut self) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        // We need to make a deep copy of this list, but it's fine; it also serves as scratch space
        // for the search within `prepare_for_native_call`.
        let exposed: Vec<AllocId> =
            this.machine.alloc_addresses.get_mut().exposed.iter().copied().collect();
        this.prepare_for_native_call(exposed)
    }
}

impl<'tcx> MiriMachine<'tcx> {
    pub fn free_alloc_id(&mut self, dead_id: AllocId, size: Size, align: Align, kind: MemoryKind) {
        let global_state = self.alloc_addresses.get_mut();
        let rng = self.rng.get_mut();

        // We can *not* remove this from `base_addr`, since the interpreter design requires that we
        // be able to retrieve an AllocId + offset for any memory access *before* we check if the
        // access is valid. Specifically, `ptr_get_alloc` is called on each attempt at a memory
        // access to determine the allocation ID and offset -- and there can still be pointers with
        // `dead_id` that one can attempt to use for a memory access. `ptr_get_alloc` may return
        // `None` only if the pointer truly has no provenance (this ensures consistent error
        // messages).
        // However, we *can* remove it from `int_to_ptr_map`, since any wildcard pointers that exist
        // can no longer actually be accessing that address. This ensures `alloc_id_from_addr` never
        // returns a dead allocation.
        // To avoid a linear scan we first look up the address in `base_addr`, and then find it in
        // `int_to_ptr_map`.
        let addr = *global_state.base_addr.get(&dead_id).unwrap();
        let pos =
            global_state.int_to_ptr_map.binary_search_by_key(&addr, |(addr, _)| *addr).unwrap();
        let removed = global_state.int_to_ptr_map.remove(pos);
        assert_eq!(removed, (addr, dead_id)); // double-check that we removed the right thing
        // We can also remove it from `exposed`, since this allocation can anyway not be returned by
        // `alloc_id_from_addr` any more.
        global_state.exposed.remove(&dead_id);
        // Also remember this address for future reuse.
        let thread = self.threads.active_thread();
        global_state.reuse.add_addr(rng, addr, size, align, kind, thread, || {
            if let Some(data_race) = self.data_race.as_vclocks_ref() {
                data_race.release_clock(&self.threads, |clock| clock.clone())
            } else {
                VClock::default()
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_addr() {
        assert_eq!(align_addr(37, 4), 40);
        assert_eq!(align_addr(44, 4), 44);
    }
}
