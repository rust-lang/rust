/// Kernel Heap Base Address
///
/// We place this at 512 GiB (0x80_0000_0000) to keep it well above the identity map
/// (usually 0-4GiB or similar) and kernel image (usually higher half).
/// Check your specific arch memory map if unsure, but this is generally safe in 64-bit higher half.
/// Actually, usually kernel is at -2GiB or similar.
/// Let's put the heap at a dedicated range.
///
/// For x86_64, canonical hole is large.
/// We'll use 0xFFFF_9000_0000_0000 as a starting point for heap?
/// Wait, let's just pick a safe range.
///
/// Common x86_64 higher half: 0xFFFF_8000_0000_0000 is often direct map.
/// Kernel image often at 0xFFFF_FFFF_8000_0000.
///
/// Let's stick the heap at 0xFFFF_A000_0000_0000 for now.
/// This gives us plenty of room.
pub const KHEAP_BASE: u64 = 0xFFFF_A000_0000_0000;

/// 256 MiB Initial Reserved Size
pub const KHEAP_SIZE: u64 = 256 * 1024 * 1024;

/// Growth increment: 16 pages (64 KiB)
pub const KHEAP_GROW_PAGES: u64 = 16;
