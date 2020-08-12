//! FIXME: write short doc here

use crate::{Subtree, TokenTree};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct EntryId(usize);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct EntryPtr(EntryId, usize);

/// Internal type which is used instead of `TokenTree` to represent a token tree
/// within a `TokenBuffer`.
#[derive(Debug)]
enum Entry<'t> {
    // Mimicking types from proc-macro.
    Subtree(&'t TokenTree, EntryId),
    Leaf(&'t TokenTree),
    // End entries contain a pointer to the entry from the containing
    // token tree, or None if this is the outermost level.
    End(Option<EntryPtr>),
}

/// A token tree buffer
/// The safe version of `syn` [`TokenBuffer`](https://github.com/dtolnay/syn/blob/6533607f91686545cb034d2838beea338d9d0742/src/buffer.rs#L41)
#[derive(Debug)]
pub struct TokenBuffer<'t> {
    buffers: Vec<Box<[Entry<'t>]>>,
}

impl<'t> TokenBuffer<'t> {
    pub fn new(tokens: &'t [TokenTree]) -> TokenBuffer<'t> {
        let mut buffers = vec![];

        let idx = TokenBuffer::new_inner(tokens, &mut buffers, None);
        assert_eq!(idx, 0);

        TokenBuffer { buffers }
    }

    fn new_inner(
        tokens: &'t [TokenTree],
        buffers: &mut Vec<Box<[Entry<'t>]>>,
        next: Option<EntryPtr>,
    ) -> usize {
        // Must contain everything in tokens and then the Entry::End
        let start_capacity = tokens.len() + 1;
        let mut entries = Vec::with_capacity(start_capacity);
        let mut children = vec![];

        for (idx, tt) in tokens.iter().enumerate() {
            match tt {
                TokenTree::Leaf(_) => {
                    entries.push(Entry::Leaf(tt));
                }
                TokenTree::Subtree(subtree) => {
                    entries.push(Entry::End(None));
                    children.push((idx, (subtree, tt)));
                }
            }
        }

        entries.push(Entry::End(next));
        let res = buffers.len();
        buffers.push(entries.into_boxed_slice());

        for (child_idx, (subtree, tt)) in children {
            let idx = TokenBuffer::new_inner(
                &subtree.token_trees,
                buffers,
                Some(EntryPtr(EntryId(res), child_idx + 1)),
            );
            buffers[res].as_mut()[child_idx] = Entry::Subtree(tt, EntryId(idx));
        }

        res
    }

    /// Creates a cursor referencing the first token in the buffer and able to
    /// traverse until the end of the buffer.
    pub fn begin(&self) -> Cursor {
        Cursor::create(self, EntryPtr(EntryId(0), 0))
    }

    fn entry(&self, ptr: &EntryPtr) -> Option<&Entry> {
        let id = ptr.0;
        self.buffers[id.0].get(ptr.1)
    }
}

/// A safe version of `Cursor` from `syn` crate https://github.com/dtolnay/syn/blob/6533607f91686545cb034d2838beea338d9d0742/src/buffer.rs#L125
#[derive(Copy, Clone, Debug)]
pub struct Cursor<'a> {
    buffer: &'a TokenBuffer<'a>,
    ptr: EntryPtr,
}

impl<'a> PartialEq for Cursor<'a> {
    fn eq(&self, other: &Cursor) -> bool {
        self.ptr == other.ptr && std::ptr::eq(self.buffer, other.buffer)
    }
}

impl<'a> Eq for Cursor<'a> {}

impl<'a> Cursor<'a> {
    /// Check whether it is eof
    pub fn eof(self) -> bool {
        matches!(self.buffer.entry(&self.ptr), None | Some(Entry::End(None)))
    }

    /// If the cursor is pointing at the end of a subtree, returns
    /// the parent subtree
    pub fn end(self) -> Option<&'a Subtree> {
        match self.entry() {
            Some(Entry::End(Some(ptr))) => {
                let idx = ptr.1;
                if let Some(Entry::Subtree(TokenTree::Subtree(subtree), _)) =
                    self.buffer.entry(&EntryPtr(ptr.0, idx - 1))
                {
                    return Some(subtree);
                }

                None
            }
            _ => None,
        }
    }

    fn entry(self) -> Option<&'a Entry<'a>> {
        self.buffer.entry(&self.ptr)
    }

    /// If the cursor is pointing at a `Subtree`, returns
    /// a cursor into that subtree
    pub fn subtree(self) -> Option<Cursor<'a>> {
        match self.entry() {
            Some(Entry::Subtree(_, entry_id)) => {
                Some(Cursor::create(self.buffer, EntryPtr(*entry_id, 0)))
            }
            _ => None,
        }
    }

    /// If the cursor is pointing at a `TokenTree`, returns it
    pub fn token_tree(self) -> Option<&'a TokenTree> {
        match self.entry() {
            Some(Entry::Leaf(tt)) => Some(tt),
            Some(Entry::Subtree(tt, _)) => Some(tt),
            Some(Entry::End(_)) => None,
            None => None,
        }
    }

    fn create(buffer: &'a TokenBuffer, ptr: EntryPtr) -> Cursor<'a> {
        Cursor { buffer, ptr }
    }

    /// Bump the cursor
    pub fn bump(self) -> Cursor<'a> {
        if let Some(Entry::End(exit)) = self.buffer.entry(&self.ptr) {
            if let Some(exit) = exit {
                Cursor::create(self.buffer, *exit)
            } else {
                self
            }
        } else {
            Cursor::create(self.buffer, EntryPtr(self.ptr.0, self.ptr.1 + 1))
        }
    }

    /// Bump the cursor, if it is a subtree, returns
    /// a cursor into that subtree
    pub fn bump_subtree(self) -> Cursor<'a> {
        match self.entry() {
            Some(Entry::Subtree(_, _)) => self.subtree().unwrap(),
            _ => self.bump(),
        }
    }

    /// Check whether it is a top level
    pub fn is_root(&self) -> bool {
        let entry_id = self.ptr.0;
        entry_id.0 == 0
    }
}
