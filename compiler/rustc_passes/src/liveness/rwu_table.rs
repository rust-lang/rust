use std::iter;

use crate::liveness::{LiveNode, Variable};

#[derive(Clone, Copy)]
pub(super) struct RWU {
    pub(super) reader: bool,
    pub(super) writer: bool,
    pub(super) used: bool,
}

/// Conceptually, this is like a `Vec<Vec<RWU>>`. But the number of
/// RWU's can get very large, so it uses a more compact representation.
pub(super) struct RWUTable {
    /// Total number of live nodes.
    live_nodes: usize,
    /// Total number of variables.
    vars: usize,

    /// A compressed representation of `RWU`s.
    ///
    /// Each word represents 2 different `RWU`s packed together. Each packed RWU
    /// is stored in 4 bits: a reader bit, a writer bit, a used bit and a
    /// padding bit.
    ///
    /// The data for each live node is contiguous and starts at a word boundary,
    /// so there might be an unused space left.
    words: Vec<u8>,
    /// Number of words per each live node.
    live_node_words: usize,
}

impl RWUTable {
    const RWU_READER: u8 = 0b0001;
    const RWU_WRITER: u8 = 0b0010;
    const RWU_USED: u8 = 0b0100;
    const RWU_MASK: u8 = 0b1111;

    /// Size of packed RWU in bits.
    const RWU_BITS: usize = 4;
    /// Size of a word in bits.
    const WORD_BITS: usize = size_of::<u8>() * 8;
    /// Number of packed RWUs that fit into a single word.
    const WORD_RWU_COUNT: usize = Self::WORD_BITS / Self::RWU_BITS;

    pub(super) fn new(live_nodes: usize, vars: usize) -> RWUTable {
        let live_node_words = vars.div_ceil(Self::WORD_RWU_COUNT);
        Self { live_nodes, vars, live_node_words, words: vec![0u8; live_node_words * live_nodes] }
    }

    fn word_and_shift(&self, ln: LiveNode, var: Variable) -> (usize, u32) {
        assert!(ln.index() < self.live_nodes);
        assert!(var.index() < self.vars);

        let var = var.index();
        let word = var / Self::WORD_RWU_COUNT;
        let shift = Self::RWU_BITS * (var % Self::WORD_RWU_COUNT);
        (ln.index() * self.live_node_words + word, shift as u32)
    }

    fn pick2_rows_mut(&mut self, a: LiveNode, b: LiveNode) -> (&mut [u8], &mut [u8]) {
        assert!(a.index() < self.live_nodes);
        assert!(b.index() < self.live_nodes);
        assert!(a != b);

        let a_start = a.index() * self.live_node_words;
        let b_start = b.index() * self.live_node_words;

        unsafe {
            let ptr = self.words.as_mut_ptr();
            (
                std::slice::from_raw_parts_mut(ptr.add(a_start), self.live_node_words),
                std::slice::from_raw_parts_mut(ptr.add(b_start), self.live_node_words),
            )
        }
    }

    pub(super) fn copy(&mut self, dst: LiveNode, src: LiveNode) {
        if dst == src {
            return;
        }

        let (dst_row, src_row) = self.pick2_rows_mut(dst, src);
        dst_row.copy_from_slice(src_row);
    }

    /// Sets `dst` to the union of `dst` and `src`, returns true if `dst` was
    /// changed.
    pub(super) fn union(&mut self, dst: LiveNode, src: LiveNode) -> bool {
        if dst == src {
            return false;
        }

        let mut changed = false;
        let (dst_row, src_row) = self.pick2_rows_mut(dst, src);
        for (dst_word, src_word) in iter::zip(dst_row, &*src_row) {
            let old = *dst_word;
            let new = *dst_word | src_word;
            *dst_word = new;
            changed |= old != new;
        }
        changed
    }

    pub(super) fn get_reader(&self, ln: LiveNode, var: Variable) -> bool {
        let (word, shift) = self.word_and_shift(ln, var);
        (self.words[word] >> shift) & Self::RWU_READER != 0
    }

    pub(super) fn get_writer(&self, ln: LiveNode, var: Variable) -> bool {
        let (word, shift) = self.word_and_shift(ln, var);
        (self.words[word] >> shift) & Self::RWU_WRITER != 0
    }

    pub(super) fn get_used(&self, ln: LiveNode, var: Variable) -> bool {
        let (word, shift) = self.word_and_shift(ln, var);
        (self.words[word] >> shift) & Self::RWU_USED != 0
    }

    pub(super) fn get(&self, ln: LiveNode, var: Variable) -> RWU {
        let (word, shift) = self.word_and_shift(ln, var);
        let rwu_packed = self.words[word] >> shift;
        RWU {
            reader: rwu_packed & Self::RWU_READER != 0,
            writer: rwu_packed & Self::RWU_WRITER != 0,
            used: rwu_packed & Self::RWU_USED != 0,
        }
    }

    pub(super) fn set(&mut self, ln: LiveNode, var: Variable, rwu: RWU) {
        let mut packed = 0;
        if rwu.reader {
            packed |= Self::RWU_READER;
        }
        if rwu.writer {
            packed |= Self::RWU_WRITER;
        }
        if rwu.used {
            packed |= Self::RWU_USED;
        }

        let (word, shift) = self.word_and_shift(ln, var);
        let word = &mut self.words[word];
        *word = (*word & !(Self::RWU_MASK << shift)) | (packed << shift)
    }
}
