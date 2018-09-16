use std::collections::HashSet;
use ra_analysis::FileId;

pub struct Subscriptions {
    subs: HashSet<FileId>,
}

impl Subscriptions {
    pub fn new() -> Subscriptions {
        Subscriptions { subs: HashSet::new() }
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
