use alloc::vec;
use alloc::vec::Vec;

pub struct Bitset {
    bits: Vec<u64>,
    size: usize,
}

impl Bitset {
    pub fn new(size: usize) -> Self {
        let words = (size + 63) / 64;
        Self {
            bits: vec![0; words],
            size,
        }
    }

    pub fn set(&mut self, index: usize) {
        if index < self.size {
            self.bits[index / 64] |= 1 << (index % 64);
        }
    }

    pub fn clear(&mut self, index: usize) {
        if index < self.size {
            self.bits[index / 64] &= !(1 << (index % 64));
        }
    }

    pub fn test(&self, index: usize) -> bool {
        if index < self.size {
            (self.bits[index / 64] & (1 << (index % 64))) != 0
        } else {
            false
        }
    }

    pub fn find_first_zero(&self) -> Option<usize> {
        for (i, &word) in self.bits.iter().enumerate() {
            if word != !0 {
                let first_zero = (!word).trailing_zeros();
                let index = i * 64 + first_zero as usize;
                if index < self.size {
                    return Some(index);
                }
            }
        }
        None
    }

    pub fn set_range(&mut self, start: usize, len: usize) {
        for i in start..(start + len) {
            self.set(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_range() {
        let mut bs = Bitset::new(100);
        bs.set_range(10, 10); // 10..20

        assert!(!bs.test(9));
        for i in 10..20 {
            assert!(bs.test(i));
        }
        assert!(!bs.test(20));
    }

    #[test]
    fn test_set_clear_test() {
        let mut bs = Bitset::new(100);
        assert!(!bs.test(10));
        bs.set(10);
        assert!(bs.test(10));
        bs.clear(10);
        assert!(!bs.test(10));
    }

    #[test]
    fn test_find_first_zero() {
        let mut bs = Bitset::new(128);
        assert_eq!(bs.find_first_zero(), Some(0));
        bs.set(0);
        assert_eq!(bs.find_first_zero(), Some(1));

        // Fill first word
        for i in 0..64 {
            bs.set(i);
        }
        assert_eq!(bs.find_first_zero(), Some(64));
    }

    #[test]
    fn test_full() {
        let mut bs = Bitset::new(10);
        for i in 0..10 {
            bs.set(i);
        }
        assert_eq!(bs.find_first_zero(), None);
    }
}
