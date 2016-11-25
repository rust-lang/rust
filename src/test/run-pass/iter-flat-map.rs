// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test .flat_map()'s size hint.
//
// Test that .flat_map() does not call the base iterator's next or next_back
// after they have returned None once.

// FinishCount is a range iterator that counts the number of times that next or
// next_back returned None.
struct FinishCount<'a> {
    start: usize,
    end: usize,
    finish_count: &'a mut usize,
}

impl<'a> FinishCount<'a> {
    fn new(start: usize, end: usize, count: &'a mut usize) -> Self {
        debug_assert!(start <= end);
        FinishCount {
            start: start,
            end: end,
            finish_count: count,
        }
    }
}

impl<'a> Iterator for FinishCount<'a> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.start == self.end {
            *self.finish_count += 1;
            return None;
        }

        let x = self.start;
        self.start += 1;
        Some(x)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = (self.end - self.start) as usize;
        (len, Some(len))
    }
}

impl<'a> DoubleEndedIterator for FinishCount<'a> {
    fn next_back(&mut self) -> Option<usize> {
        if self.start == self.end {
            *self.finish_count += 1;
            return None;
        }

        self.end -= 1;
        Some(self.end)
    }
}

fn main() {
    // Try a 3 × 2 cartesian product (6 elements)
    // The iterator permits 6 .next() or .next_back() calls before returning
    // None on the 7th.
    // try all 2**7 == 128 possible permutations of next and next_back calls.

    let outer_len = 3;
    let inner_len = 2;
    let len = inner_len * outer_len;
    let combinations = 1u64 << (len + 1);

    // Produce the sequence (0, 0), (0, 1), .. (1, 0), (1, 1) .. etc
    let answer = (0..outer_len).flat_map(|i| (0..inner_len).map(move |j| (i, j)))
                               .collect::<Vec<_>>();

    let mut front_elem = Vec::new();
    let mut back_elem = Vec::new();
    for combination in 0..combinations {
        front_elem.clear();
        back_elem.clear();
        let mut finish_count = 0;
        {
            let mut iter = FinishCount::new(0, outer_len, &mut finish_count)
                            .flat_map(|i| (0..inner_len).map(move |j| (i, j)));
            for bit in 0..(len + 1) {
                // test size hint
                let (low, opt_hi) = iter.size_hint();
                assert!(low <= len - bit, "flat_map overestimates lower bound");
                if let Some(hi) = opt_hi {
                    assert!(hi >= len - bit, "flat_map underestimates upper bound");
                }

                // take the next element
                // bit 1 => next, 0 => next_back
                if combination & (1 << bit) != 0 {
                    if let Some(elt) = iter.next() {
                        front_elem.push(elt);
                    } else {
                        assert_eq!(bit, len);
                    }
                } else {
                    if let Some(elt) = iter.next_back() {
                        back_elem.push(elt);
                    } else {
                        assert_eq!(bit, len);
                    }
                }
            }
        }

        back_elem.reverse();
        if !Iterator::eq(answer.iter(), front_elem.iter().chain(&back_elem)) {
            panic!("did not produce the same sequence: expected={:?}, got={:?}{:?}",
                   answer, front_elem, back_elem);
        }
        assert_eq!(finish_count, 1,
                   "Combination {:b} called inner iter's .next() after it returned None",
                   combination);
    }
}
