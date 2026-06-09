//@ normalize-stderr-test: "(\n)ALLOC \(.*\) \{\n(.*\n)*\}(\n)" -> "${1}ALLOC DUMP${3}"
//@ normalize-stderr-test: "\[0x[0-9a-z]..0x[0-9a-z]\]" -> "[0xX..0xY]"

use std::mem;

// If this is `None`, the metadata becomes padding.
type T = Option<&'static str>;

fn main() {
    unsafe {
        let mut p: mem::MaybeUninit<T> = mem::MaybeUninit::zeroed();
        // The copy when `T` is returned from `transmute` should destroy padding
        // (even when we use `write_unaligned`, which under the hood uses an untyped copy).
        p.as_mut_ptr().write_unaligned(mem::transmute((0usize, 0usize)));
        // Null epresents `None`.
        assert!(matches!(*p.as_ptr(), None));

        // The second part, with the length, becomes padding.
        let c = &p as *const _ as *const u8;
        // Read a padding byte.
        let _val = *c.add(mem::size_of::<*const u8>());
        //~^ERROR: uninitialized
    }
}
