use std::marker::PhantomData;

use rustc_index::vec::Idx;

use ena::undo_log::{Rollback, UndoLogs};

#[derive(Copy, Clone, Debug)]
enum UndoInner {
    Add,
    Drain { index: usize, offset: usize },
}

#[derive(Copy, Clone, Debug)]
pub struct Undo<I>(UndoInner, PhantomData<I>);

/// Tracks which indices have been modified and allows watchers to registered and notified of these
/// changes.
#[derive(Clone, Debug)]
pub struct ModifiedSet<T: Idx> {
    modified: Vec<T>,
    offsets: Vec<usize>,
}

impl<T: Idx> Default for ModifiedSet<T> {
    fn default() -> Self {
        Self { modified: Default::default(), offsets: Vec::new() }
    }
}

impl<T: Idx> ModifiedSet<T> {
    /// Creates a new `ModifiedSet`
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks `index` as "modified". A subsequent call to `drain` will notify the callback with
    /// `index`
    pub fn set(&mut self, undo_log: &mut impl UndoLogs<Undo<T>>, index: T) {
        self.modified.push(index);
        undo_log.push(Undo(UndoInner::Add, PhantomData));
    }

    /// Calls `f` with all the indices that have been modified since the last call to
    /// `notify_watcher`
    pub fn notify_watcher(
        &mut self,
        undo_log: &mut impl UndoLogs<Undo<T>>,
        watcher_offset: &Offset<T>,
        mut f: impl FnMut(T),
    ) {
        let offset = &mut self.offsets[watcher_offset.index];
        if *offset < self.modified.len() {
            for &index in &self.modified[*offset..] {
                f(index);
            }
            undo_log.push(Undo(
                UndoInner::Drain { index: watcher_offset.index, offset: *offset },
                PhantomData,
            ));
            *offset = self.modified.len();
        }
    }

    /// Clears the set of all modifications that have been drained by all watchers
    pub fn clear(&mut self) {
        let min = self.offsets.iter().copied().min().unwrap_or_else(|| self.modified.len());
        self.modified.drain(..min);
        for offset in &mut self.offsets {
            *offset -= min;
        }
    }

    /// Registers a new watcher on this set.
    ///
    /// NOTE: Watchers must be removed in the reverse order that they were registered
    pub fn register(&mut self) -> Offset<T> {
        let index = self.offsets.len();
        self.offsets.push(self.modified.len());
        Offset { index, _marker: PhantomData }
    }

    /// De-registers a watcher on this set.
    ///
    /// NOTE: Watchers must be removed in the reverse order that they were registered
    pub fn deregister(&mut self, offset: Offset<T>) {
        assert_eq!(
            offset.index,
            self.offsets.len() - 1,
            "Watchers must be removed in the reverse order that they were registered"
        );
        self.offsets.pop();
        std::mem::forget(offset);
    }
}

impl<I: Idx> Rollback<Undo<I>> for ModifiedSet<I> {
    fn reverse(&mut self, undo: Undo<I>) {
        match undo.0 {
            UndoInner::Add => {
                self.modified.pop();
            }
            UndoInner::Drain { index, offset } => {
                if let Some(o) = self.offsets.get_mut(index) {
                    *o = offset;
                }
            }
        }
    }
}

/// A registered offset into a `ModifiedSet`. Tracks how much a watcher has seen so far to avoid
/// being notified of the same event twice.
#[must_use]
pub struct Offset<T> {
    index: usize,
    _marker: PhantomData<T>,
}

impl<T> Drop for Offset<T> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            panic!("Offsets should be deregistered")
        }
    }
}

#[must_use]
#[derive(Debug)]
pub struct Snapshot<T> {
    modified_len: usize,
    undo_log_len: usize,
    _marker: PhantomData<T>,
}
