use std::cell::RefCell;

use rustc::mir::interpret::AllocId;

pub type MemoryState = RefCell<GlobalState>;

#[derive(Clone, Debug)]
pub struct GlobalState {
    /// This field is used as a map between the address of each allocation and its `AllocId`
    pub int_to_ptr_map: Vec<(u64, AllocId)>,
    pub next_base_addr: u64,
}

impl Default for GlobalState {
    // FIXME: Query the page size in the future
    fn default() -> Self {
        GlobalState {
            int_to_ptr_map: Vec::default(),
            next_base_addr: 2u64.pow(16)
        }
    }
}
