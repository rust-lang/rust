//! Keeps track of file subscriptions.

use ra_ide::FileId;
use rustc_hash::FxHashSet;

#[derive(Default, Debug)]
pub(crate) struct Subscriptions {
    subs: FxHashSet<FileId>,
}

impl Subscriptions {
    pub(crate) fn add_sub(&mut self, file_id: FileId) {
        self.subs.insert(file_id);
    }
    pub(crate) fn remove_sub(&mut self, file_id: FileId) {
        self.subs.remove(&file_id);
    }
    pub(crate) fn subscriptions(&self) -> Vec<FileId> {
        self.subs.iter().cloned().collect()
    }
}
