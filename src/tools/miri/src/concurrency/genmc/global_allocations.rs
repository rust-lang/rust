use std::collections::hash_map::Entry;
use std::sync::RwLock;

use genmc_sys::{GENMC_GLOBAL_ADDRESSES_MASK, get_global_alloc_static_mask};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rustc_const_eval::interpret::{AllocId, AllocInfo, InterpResult, interp_ok};
use rustc_data_structures::fx::FxHashMap;
use rustc_log::tracing::debug;

use crate::alloc_addresses::AddressGenerator;

#[derive(Debug)]
struct GlobalStateInner {
    /// The base address for each *global* allocation.
    base_addr: FxHashMap<AllocId, u64>,
    /// We use the same address generator that Miri uses in normal operation.
    address_generator: AddressGenerator,
    /// The address generator needs an Rng to randomize the offsets between allocations.
    /// We don't use the `MiriMachine` Rng since this is global, cross-machine state.
    rng: StdRng,
}

/// Allocator for global memory in GenMC mode.
/// Miri doesn't discover all global allocations statically like LLI does for GenMC.
/// The existing global memory allocator in GenMC doesn't support this, so we take over these allocations.
/// Global allocations need to be in a specific address range, with the lower limit given by the `GENMC_GLOBAL_ADDRESSES_MASK` constant.
///
/// Every global allocation must have the same addresses across all executions of a single program.
/// Therefore there is only 1 global allocator, and it syncs new globals across executions, even if they are explored in parallel.
#[derive(Debug)]
pub struct GlobalAllocationHandler(RwLock<GlobalStateInner>);

impl GlobalAllocationHandler {
    /// Create a new global address generator with a given max address `last_addr`
    /// (corresponding to the highest address available on the target platform, unless another limit exists).
    /// No addresses higher than this will be allocated.
    /// Will panic if the given address limit is too small to allocate any addresses.
    pub fn new(last_addr: u64) -> GlobalAllocationHandler {
        assert_eq!(GENMC_GLOBAL_ADDRESSES_MASK, get_global_alloc_static_mask());
        assert_ne!(GENMC_GLOBAL_ADDRESSES_MASK, 0);
        // FIXME(genmc): Remove if non-64bit targets are supported.
        assert!(
            GENMC_GLOBAL_ADDRESSES_MASK < last_addr,
            "only 64bit platforms are currently supported (highest address {last_addr:#x} <= minimum global address {GENMC_GLOBAL_ADDRESSES_MASK:#x})."
        );
        Self(RwLock::new(GlobalStateInner {
            base_addr: FxHashMap::default(),
            address_generator: AddressGenerator::new(GENMC_GLOBAL_ADDRESSES_MASK..last_addr),
            // FIXME(genmc): We could provide a way to changes this seed, to allow for different global addresses.
            rng: StdRng::seed_from_u64(0),
        }))
    }
}

impl GlobalStateInner {
    fn global_allocate_addr<'tcx>(
        &mut self,
        alloc_id: AllocId,
        info: AllocInfo,
    ) -> InterpResult<'tcx, u64> {
        let entry = match self.base_addr.entry(alloc_id) {
            Entry::Occupied(occupied_entry) => {
                // Looks like some other thread allocated this for us
                // between when we released the read lock and aquired the write lock,
                // so we just return that value.
                return interp_ok(*occupied_entry.get());
            }
            Entry::Vacant(vacant_entry) => vacant_entry,
        };

        // This allocation does not have a base address yet, pick or reuse one.
        // We are not in native lib mode (incompatible with GenMC mode), so we control the addresses ourselves.
        let new_addr = self.address_generator.generate(info.size, info.align, &mut self.rng)?;

        // Cache the address for future use.
        entry.insert(new_addr);

        interp_ok(new_addr)
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub(super) trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Allocate a new address for the given alloc id, or return the cached address.
    /// Each alloc id is assigned one unique allocation which will not change if this function is called again with the same alloc id.
    fn get_global_allocation_address(
        &self,
        global_allocation_handler: &GlobalAllocationHandler,
        alloc_id: AllocId,
    ) -> InterpResult<'tcx, u64> {
        let this = self.eval_context_ref();
        let info = this.get_alloc_info(alloc_id);

        let global_state = global_allocation_handler.0.read().unwrap();
        if let Some(base_addr) = global_state.base_addr.get(&alloc_id) {
            debug!(
                "GenMC: address for global with alloc id {alloc_id:?} was cached: {base_addr} == {base_addr:#x}"
            );
            return interp_ok(*base_addr);
        }

        // We need to upgrade to a write lock. `std::sync::RwLock` doesn't support this, so we drop the guard and lock again
        // Note that another thread might allocate the address while the `RwLock` is unlocked, but we handle this case in the allocation function.
        drop(global_state);
        let mut global_state = global_allocation_handler.0.write().unwrap();
        // With the write lock, we can safely allocate an address only once per `alloc_id`.
        let new_addr = global_state.global_allocate_addr(alloc_id, info)?;
        debug!("GenMC: global with alloc id {alloc_id:?} got address: {new_addr} == {new_addr:#x}");
        assert_eq!(
            GENMC_GLOBAL_ADDRESSES_MASK,
            new_addr & GENMC_GLOBAL_ADDRESSES_MASK,
            "Global address allocated outside global address space."
        );

        interp_ok(new_addr)
    }
}
