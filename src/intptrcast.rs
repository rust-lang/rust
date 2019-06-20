use std::cell::RefCell;

use rustc::mir::interpret::AllocId;

pub type MemoryState = RefCell<GlobalState>;

#[derive(Clone, Debug)]
pub struct GlobalState {
    pub vec: Vec<(u64, AllocId)>,
    pub addr: u64,
}

impl Default for GlobalState {
    fn default() -> Self {
        GlobalState {
            vec: Vec::default(),
            addr: 2u64.pow(16)
        }
    }
}
