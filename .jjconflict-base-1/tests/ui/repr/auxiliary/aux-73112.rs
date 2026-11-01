#[repr(transparent)]
pub struct PageTableEntry {
    entry: u64,
}

#[repr(align(4096))]
#[repr(C)]
pub struct PageTable {
    entries: [PageTableEntry; 512],
}
