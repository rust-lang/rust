//! Stateful iteration over token trees.
//!
//! We use this as the source of tokens for parser.
use crate::{Leaf, Subtree, TokenTree};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct EntryId(usize);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct EntryPtr(
    /// The index of the buffer containing the entry.
    EntryId,
    /// The index of the entry within the buffer.
    usize,
);

/// Internal type which is used instead of `TokenTree` to represent a token tree
/// within a `TokenBuffer`.
#[derive(Debug)]
enum Entry<'t, Span> {
    // Mimicking types from proc-macro.
    Subtree(Option<&'t TokenTree<Span>>, &'t Subtree<Span>, EntryId),
    Leaf(&'t TokenTree<Span>),
    /// End entries contain a pointer to the entry from the containing
    /// token tree, or [`None`] if this is the outermost level.
    End(Option<EntryPtr>),
}

/// A token tree buffer
/// The safe version of `syn` [`TokenBuffer`](https://github.com/dtolnay/syn/blob/6533607f91686545cb034d2838beea338d9d0742/src/buffer.rs#L41)
#[derive(Debug)]
pub struct TokenBuffer<'t, Span> {
    buffers: Vec<Box<[Entry<'t, Span>]>>,
}

trait TokenList<'a, Span> {
    fn entries(
        &self,
    ) -> (Vec<(usize, (&'a Subtree<Span>, Option<&'a TokenTree<Span>>))>, Vec<Entry<'a, Span>>);
}

impl<'a, Span> TokenList<'a, Span> for &'a [TokenTree<Span>] {
    fn entries(
        &self,
    ) -> (Vec<(usize, (&'a Subtree<Span>, Option<&'a TokenTree<Span>>))>, Vec<Entry<'a, Span>>)
    {
        // Must contain everything in tokens and then the Entry::End
        let start_capacity = self.len() + 1;
        let mut entries = Vec::with_capacity(start_capacity);
        let mut children = vec![];
        for (idx, tt) in self.iter().enumerate() {
            match tt {
                TokenTree::Leaf(_) => {
                    entries.push(Entry::Leaf(tt));
                }
                TokenTree::Subtree(subtree) => {
                    entries.push(Entry::End(None));
                    children.push((idx, (subtree, Some(tt))));
                }
            }
        }
        (children, entries)
    }
}

impl<'a, Span> TokenList<'a, Span> for &'a Subtree<Span> {
    fn entries(
        &self,
    ) -> (Vec<(usize, (&'a Subtree<Span>, Option<&'a TokenTree<Span>>))>, Vec<Entry<'a, Span>>)
    {
        // Must contain everything in tokens and then the Entry::End
        let mut entries = vec![];
        let mut children = vec![];
        entries.push(Entry::End(None));
        children.push((0usize, (*self, None)));
        (children, entries)
    }
}

impl<'t, Span> TokenBuffer<'t, Span> {
    pub fn from_tokens(tokens: &'t [TokenTree<Span>]) -> TokenBuffer<'t, Span> {
        Self::new(tokens)
    }

    pub fn from_subtree(subtree: &'t Subtree<Span>) -> TokenBuffer<'t, Span> {
        Self::new(subtree)
    }

    fn new<T: TokenList<'t, Span>>(tokens: T) -> TokenBuffer<'t, Span> {
        let mut buffers = vec![];
        let idx = TokenBuffer::new_inner(tokens, &mut buffers, None);
        assert_eq!(idx, 0);
        TokenBuffer { buffers }
    }

    fn new_inner<T: TokenList<'t, Span>>(
        tokens: T,
        buffers: &mut Vec<Box<[Entry<'t, Span>]>>,
        next: Option<EntryPtr>,
    ) -> usize {
        let (children, mut entries) = tokens.entries();

        entries.push(Entry::End(next));
        let res = buffers.len();
        buffers.push(entries.into_boxed_slice());

        for (child_idx, (subtree, tt)) in children {
            let idx = TokenBuffer::new_inner(
                subtree.token_trees.as_slice(),
                buffers,
                Some(EntryPtr(EntryId(res), child_idx + 1)),
            );
            buffers[res].as_mut()[child_idx] = Entry::Subtree(tt, subtree, EntryId(idx));
        }

        res
    }

    /// Creates a cursor referencing the first token in the buffer and able to
    /// traverse until the end of the buffer.
    pub fn begin(&self) -> Cursor<'_, Span> {
        Cursor::create(self, EntryPtr(EntryId(0), 0))
    }

    fn entry(&self, ptr: &EntryPtr) -> Option<&Entry<'_, Span>> {
        let id = ptr.0;
        self.buffers[id.0].get(ptr.1)
    }
}

#[derive(Debug)]
pub enum TokenTreeRef<'a, Span> {
    Subtree(&'a Subtree<Span>, Option<&'a TokenTree<Span>>),
    Leaf(&'a Leaf<Span>, &'a TokenTree<Span>),
}

impl<Span: Clone> TokenTreeRef<'_, Span> {
    pub fn cloned(&self) -> TokenTree<Span> {
        match self {
            TokenTreeRef::Subtree(subtree, tt) => match tt {
                Some(it) => (*it).clone(),
                None => (*subtree).clone().into(),
            },
            TokenTreeRef::Leaf(_, tt) => (*tt).clone(),
        }
    }
}

/// A safe version of `Cursor` from `syn` crate <https://github.com/dtolnay/syn/blob/6533607f91686545cb034d2838beea338d9d0742/src/buffer.rs#L125>
#[derive(Copy, Clone, Debug)]
pub struct Cursor<'a, Span> {
    buffer: &'a TokenBuffer<'a, Span>,
    ptr: EntryPtr,
}

impl<Span> PartialEq for Cursor<'_, Span> {
    fn eq(&self, other: &Cursor<'_, Span>) -> bool {
        self.ptr == other.ptr && std::ptr::eq(self.buffer, other.buffer)
    }
}

impl<Span> Eq for Cursor<'_, Span> {}

impl<'a, Span> Cursor<'a, Span> {
    /// Check whether it is eof
    pub fn eof(self) -> bool {
        matches!(self.buffer.entry(&self.ptr), None | Some(Entry::End(None)))
    }

    /// If the cursor is pointing at the end of a subtree, returns
    /// the parent subtree
    pub fn end(self) -> Option<&'a Subtree<Span>> {
        match self.entry() {
            Some(Entry::End(Some(ptr))) => {
                let idx = ptr.1;
                if let Some(Entry::Subtree(_, subtree, _)) =
                    self.buffer.entry(&EntryPtr(ptr.0, idx - 1))
                {
                    return Some(subtree);
                }
                None
            }
            _ => None,
        }
    }

    fn entry(&self) -> Option<&'a Entry<'a, Span>> {
        self.buffer.entry(&self.ptr)
    }

    /// If the cursor is pointing at a `Subtree`, returns
    /// a cursor into that subtree
    pub fn subtree(self) -> Option<Cursor<'a, Span>> {
        match self.entry() {
            Some(Entry::Subtree(_, _, entry_id)) => {
                Some(Cursor::create(self.buffer, EntryPtr(*entry_id, 0)))
            }
            _ => None,
        }
    }

    /// If the cursor is pointing at a `TokenTree`, returns it
    pub fn token_tree(self) -> Option<TokenTreeRef<'a, Span>> {
        match self.entry() {
            Some(Entry::Leaf(tt)) => match tt {
                TokenTree::Leaf(leaf) => Some(TokenTreeRef::Leaf(leaf, tt)),
                TokenTree::Subtree(subtree) => Some(TokenTreeRef::Subtree(subtree, Some(tt))),
            },
            Some(Entry::Subtree(tt, subtree, _)) => Some(TokenTreeRef::Subtree(subtree, *tt)),
            Some(Entry::End(_)) | None => None,
        }
    }

    fn create(buffer: &'a TokenBuffer<'_, Span>, ptr: EntryPtr) -> Cursor<'a, Span> {
        Cursor { buffer, ptr }
    }

    /// Bump the cursor
    pub fn bump(self) -> Cursor<'a, Span> {
        if let Some(Entry::End(exit)) = self.buffer.entry(&self.ptr) {
            match exit {
                Some(exit) => Cursor::create(self.buffer, *exit),
                None => self,
            }
        } else {
            Cursor::create(self.buffer, EntryPtr(self.ptr.0, self.ptr.1 + 1))
        }
    }

    /// Bump the cursor, if it is a subtree, returns
    /// a cursor into that subtree
    pub fn bump_subtree(self) -> Cursor<'a, Span> {
        match self.entry() {
            Some(&Entry::Subtree(_, _, entry_id)) => {
                Cursor::create(self.buffer, EntryPtr(entry_id, 0))
            }
            Some(Entry::End(exit)) => match exit {
                Some(exit) => Cursor::create(self.buffer, *exit),
                None => self,
            },
            _ => Cursor::create(self.buffer, EntryPtr(self.ptr.0, self.ptr.1 + 1)),
        }
    }

    /// Check whether it is a top level
    pub fn is_root(&self) -> bool {
        let entry_id = self.ptr.0;
        entry_id.0 == 0
    }
}
