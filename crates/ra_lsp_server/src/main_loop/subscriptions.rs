use rustc_hash::FxHashSet;
use ra_analysis::FileId;

pub struct Subscriptions {
    subs: FxHashSet<FileId>,
}

impl Subscriptions {
    pub fn new() -> Subscriptions {
        Subscriptions { subs: FxHashSet::default() }
    }
    pub fn add_sub(&mut self, file_id: FileId) {
        self.subs.insert(file_id);
    }
    pub fn remove_sub(&mut self, file_id: FileId) {
        self.subs.remove(&file_id);
    }
    pub fn subscriptions(&self) -> Vec<FileId> {
        self.subs.iter().cloned().collect()
    }
}
