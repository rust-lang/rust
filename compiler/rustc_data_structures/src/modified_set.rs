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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, undo_log: &mut impl UndoLogs<Undo<T>>, index: T) {
        self.modified.push(index);
        undo_log.push(Undo(UndoInner::Add, PhantomData));
    }

    pub fn drain(
        &mut self,
        undo_log: &mut impl UndoLogs<Undo<T>>,
        index: &Offset<T>,
        mut f: impl FnMut(T) -> bool,
    ) {
        let offset = &mut self.offsets[index.index];
        if *offset < self.modified.len() {
            for &index in &self.modified[*offset..] {
                f(index);
            }
            undo_log
                .push(Undo(UndoInner::Drain { index: index.index, offset: *offset }, PhantomData));
            *offset = self.modified.len();
        }
    }

    pub fn clear(&mut self) {
        let min = self.offsets.iter().copied().min().unwrap_or_else(|| self.modified.len());
        self.modified.drain(..min);
        for offset in &mut self.offsets {
            *offset -= min;
        }
    }

    pub fn register(&mut self) -> Offset<T> {
        let index = self.offsets.len();
        self.offsets.push(self.modified.len());
        Offset { index, _marker: PhantomData }
    }

    pub fn deregister(&mut self, offset: Offset<T>) {
        assert_eq!(offset.index, self.offsets.len() - 1);
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
