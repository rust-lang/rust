#![allow(unused_imports)]
use crate::{PhysRange, PhysRangeKind};
use alloc::vec::Vec;

pub fn normalize(mut ranges: Vec<PhysRange>) -> Vec<PhysRange> {
    // 1. Sort by start address
    ranges.sort_unstable_by_key(|r| r.start);

    // 2. Coalesce adjacent usable ranges
    let mut out = Vec::new();
    if ranges.is_empty() {
        return out;
    }

    let mut current = ranges[0];
    for next in ranges.into_iter().skip(1) {
        if current.kind == next.kind && current.end == next.start {
            current.end = next.end;
        } else {
            out.push(current);
            current = next;
        }
    }
    out.push(current);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_sorting_and_coalescing() {
        let input = vec![
            PhysRange {
                start: 100,
                end: 200,
                kind: PhysRangeKind::Usable,
            },
            PhysRange {
                start: 0,
                end: 100,
                kind: PhysRangeKind::Usable,
            },
            PhysRange {
                start: 200,
                end: 300,
                kind: PhysRangeKind::Reserved,
            },
        ];
        let normalized = normalize(input);

        // Sorted: [0, 100) Usable, [100, 200) Usable, [200, 300) Reserved
        // Coalesced Usable: [0, 200) Usable, [200, 300) Reserved

        assert_eq!(normalized.len(), 2);
        assert_eq!(normalized[0].start, 0);
        assert_eq!(normalized[0].end, 200);
        assert_eq!(normalized[0].kind, PhysRangeKind::Usable);

        assert_eq!(normalized[1].start, 200);
        assert_eq!(normalized[1].end, 300);
        assert_eq!(normalized[1].kind, PhysRangeKind::Reserved);
    }
}
