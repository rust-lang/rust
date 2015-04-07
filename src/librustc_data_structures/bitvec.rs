use std::iter;

/// A very simple BitVector type.
pub struct BitVector {
    data: Vec<u64>
}

impl BitVector {
    pub fn new(num_bits: usize) -> BitVector {
        let num_words = (num_bits + 63) / 64;
        BitVector { data: iter::repeat(0).take(num_words).collect() }
    }

    fn word_mask(&self, bit: usize) -> (usize, u64) {
        let word = bit / 64;
        let mask = 1 << (bit % 64);
        (word, mask)
    }

    pub fn contains(&self, bit: usize) -> bool {
        let (word, mask) = self.word_mask(bit);
        (self.data[word] & mask) != 0
    }

    pub fn insert(&mut self, bit: usize) -> bool {
        let (word, mask) = self.word_mask(bit);
        let data = &mut self.data[word];
        let value = *data;
        *data = value | mask;
        (value | mask) != value
    }
}
