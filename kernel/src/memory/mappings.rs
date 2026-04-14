use abi::vm::{VmProt, VmRegionInfo};
use alloc::vec::Vec;
use core::cmp;

#[derive(Debug, Clone, Default)]
pub struct MappingList {
    pub regions: Vec<VmRegionInfo>,
}

impl MappingList {
    pub fn new() -> Self {
        MappingList {
            regions: Vec::new(),
        }
    }

    /// Insert a new region, merging with adjacent regions if possible.
    pub fn insert(&mut self, new_region: VmRegionInfo) {
        // Ensure start < end
        if new_region.start >= new_region.end {
            return;
        }

        // Find insertion point
        let mut i = 0;
        while i < self.regions.len() {
            if self.regions[i].start > new_region.start {
                break;
            }
            i += 1;
        }

        // Insert temporarily
        self.regions.insert(i, new_region);

        // Merge passes
        // We only need to check i-1 and i (now i+1)

        // Check merge with previous
        if i > 0 {
            if self.can_merge(&self.regions[i - 1], &self.regions[i]) {
                self.regions[i - 1].end = self.regions[i].end;
                self.regions.remove(i);
                i -= 1; // Stay at merged index to check next
            }
        }

        // Check merge with next
        if i + 1 < self.regions.len() {
            if self.can_merge(&self.regions[i], &self.regions[i + 1]) {
                self.regions[i].end = self.regions[i + 1].end;
                self.regions.remove(i + 1);
            }
        }
    }

    fn can_merge(&self, left: &VmRegionInfo, right: &VmRegionInfo) -> bool {
        if left.end != right.start {
            return false;
        }
        if left.prot != right.prot {
            return false;
        }
        if left.flags != right.flags {
            return false;
        }
        if left.backing_kind != right.backing_kind {
            return false;
        }
        true
    }

    /// Remove a range of addresses from the mappings.
    /// Returns the list of physical/virtual ranges that were actually removed.
    /// This handles splitting regions.
    pub fn remove(&mut self, start: usize, len: usize) -> Vec<(usize, usize)> {
        let end = start + len;
        let mut removed_ranges = Vec::new();

        let mut i = 0;
        while i < self.regions.len() {
            let r = &mut self.regions[i];

            // Case 1: No overlap
            if r.end <= start || r.start >= end {
                i += 1;
                continue;
            }

            // Overlap detected

            // Calculate overlap range
            let overlap_start = cmp::max(r.start, start);
            let overlap_end = cmp::min(r.end, end);
            removed_ranges.push((overlap_start, overlap_end));

            // Case 2: Region is fully contained in removal range -> delete region
            if r.start >= start && r.end <= end {
                self.regions.remove(i);
                // Don't increment i, next element is now at i
                continue;
            }

            // Case 3: Removal range is fully contained in region -> split region
            if r.start < start && r.end > end {
                let right_start = end;
                let right_end = r.end;

                // Truncate current (left)
                r.end = start;

                // Insert new (right)
                let new_region = VmRegionInfo {
                    start: right_start,
                    end: right_end,
                    prot: r.prot,
                    flags: r.flags,
                    backing_kind: r.backing_kind,
                    _reserved: [0; 7],
                };
                self.regions.insert(i + 1, new_region);

                i += 2; // Skip the newly inserted one
                continue;
            }

            // Case 4: Overlap at end of region (Head trim)
            // r.start < start, but r.end <= end (and > start)
            if r.start < start && r.end <= end {
                r.end = start;
                i += 1;
                continue;
            }

            // Case 5: Overlap at start of region (Tail trim)
            // r.start >= start, but r.end > end
            if r.start >= start && r.end > end {
                r.start = end;
                i += 1;
                continue;
            }

            i += 1;
        }

        removed_ranges
    }

    /// Split any region that contains `addr` at that address.
    pub fn split_at(&mut self, addr: usize) {
        let mut i = 0;
        while i < self.regions.len() {
            let r = self.regions[i];
            if addr > r.start && addr < r.end {
                // Split r into [r.start, addr) and [addr, r.end)
                self.regions[i].end = addr;
                let new_region = VmRegionInfo {
                    start: addr,
                    end: r.end,
                    prot: r.prot,
                    flags: r.flags,
                    backing_kind: r.backing_kind,
                    _reserved: [0; 7],
                };
                self.regions.insert(i + 1, new_region);
                break; // A single address can only split one region in a disjoint list
            }
            i += 1;
        }
    }

    /// Update protection for a range of addresses.
    /// This assumes the range is fully covered by mappings.
    pub fn protect(&mut self, start: usize, len: usize, new_prot: VmProt) {
        let end = start + len;

        // 1. Split at boundaries
        self.split_at(start);
        self.split_at(end);

        // 2. Update prot for regions within [start, end)
        for r in &mut self.regions {
            if r.start >= start && r.end <= end {
                r.prot = new_prot;
            }
        }

        // 3. Merge identical neighbors
        self.merge_all();
    }

    /// Merge all adjacent regions that have identical properties.
    pub fn merge_all(&mut self) {
        let mut i = 0;
        while i + 1 < self.regions.len() {
            if self.can_merge(&self.regions[i], &self.regions[i + 1]) {
                self.regions[i].end = self.regions[i + 1].end;
                self.regions.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }

    pub fn find_at(&self, addr: usize) -> Option<VmRegionInfo> {
        // Binary search or linear scan? Linear is fine for now.
        for region in &self.regions {
            if region.start == addr {
                return Some(*region);
            }
        }
        None
    }

    pub fn check(&self, addr: usize, len: usize, write: bool) -> bool {
        if len == 0 {
            return true;
        }
        let end = match addr.checked_add(len) {
            Some(e) => e,
            None => return false,
        };

        // We need to verify that [addr, end) is FULLY covered by valid regions.
        // And check permissions.

        let mut current = addr;
        for region in &self.regions {
            if region.end <= current {
                continue;
            }
            if region.start > current {
                // Gap detected
                return false;
            }

            // Check perms
            if write && !region.prot.contains(VmProt::WRITE) {
                return false;
            }

            // region covers [current, region.end)
            current = region.end;
            if current >= end {
                return true;
            }
        }

        // If loop finished and we haven't reached end
        current >= end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_merge() {
        let mut list = MappingList::new();
        let r1 = VmRegionInfo {
            start: 0x1000,
            end: 0x2000,
            prot: VmProt::READ,
            ..Default::default()
        };
        let r2 = VmRegionInfo {
            start: 0x2000,
            end: 0x3000,
            prot: VmProt::READ,
            ..Default::default()
        };

        list.insert(r1);
        list.insert(r2);

        assert_eq!(list.regions.len(), 1);
        assert_eq!(list.regions[0].start, 0x1000);
        assert_eq!(list.regions[0].end, 0x3000);
    }

    #[test]
    fn test_remove_split() {
        let mut list = MappingList::new();
        let r1 = VmRegionInfo {
            start: 0x1000,
            end: 0x4000,
            prot: VmProt::READ,
            ..Default::default()
        };
        list.insert(r1);

        let removed = list.remove(0x2000, 0x1000);

        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0], (0x2000, 0x3000));

        assert_eq!(list.regions.len(), 2);
        assert_eq!(list.regions[0].start, 0x1000);
        assert_eq!(list.regions[0].end, 0x2000);
        assert_eq!(list.regions[1].start, 0x3000);
        assert_eq!(list.regions[1].end, 0x4000);
    }

    #[test]
    fn test_check() {
        let mut list = MappingList::new();
        let r1 = VmRegionInfo {
            start: 0x1000,
            end: 0x2000,
            prot: VmProt::READ,
            ..Default::default()
        };
        let r2 = VmRegionInfo {
            start: 0x2000,
            end: 0x3000,
            prot: VmProt::READ | VmProt::WRITE,
            ..Default::default()
        };
        list.insert(r1);
        list.insert(r2);

        assert!(list.check(0x1000, 0x2000, false));
        assert!(list.check(0x1000, 0x1000, false));
        assert!(!list.check(0x1000, 0x2000, true)); // r1 not writable
        assert!(list.check(0x2000, 0x1000, true));
        assert!(list.check(0x1500, 0x1000, false)); // Spans both, read ok
        assert!(!list.check(0x1500, 0x1000, true)); // Spans both, first part not writable
        assert!(!list.check(0x0500, 0x1000, false)); // Start before
        assert!(!list.check(0x2500, 0x1000, false)); // End after
    }

    #[test]
    fn test_protect() {
        let mut list = MappingList::new();
        let r1 = VmRegionInfo {
            start: 0x1000,
            end: 0x4000,
            prot: VmProt::READ | VmProt::WRITE,
            ..Default::default()
        };
        list.insert(r1);

        // Protect middle page
        list.protect(0x2000, 0x1000, VmProt::READ);

        assert_eq!(list.regions.len(), 3);
        assert_eq!(list.regions[0].start, 0x1000);
        assert_eq!(list.regions[0].end, 0x2000);
        assert_eq!(list.regions[0].prot, VmProt::READ | VmProt::WRITE);

        assert_eq!(list.regions[1].start, 0x2000);
        assert_eq!(list.regions[1].end, 0x3000);
        assert_eq!(list.regions[1].prot, VmProt::READ);

        assert_eq!(list.regions[2].start, 0x3000);
        assert_eq!(list.regions[2].end, 0x4000);
        assert_eq!(list.regions[2].prot, VmProt::READ | VmProt::WRITE);

        // Protect entire range back to RW
        list.protect(0x1000, 0x3000, VmProt::READ | VmProt::WRITE);
        assert_eq!(list.regions.len(), 1);
        assert_eq!(list.regions[0].start, 0x1000);
        assert_eq!(list.regions[0].end, 0x4000);
        assert_eq!(list.regions[0].prot, VmProt::READ | VmProt::WRITE);
    }
}
