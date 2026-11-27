//@ revisions:rpass1 rpass2 rpass3
//@ ignore-backends: gcc

// See issue #57692.

#![allow(warnings)]

fn main() {
    #[cfg(rpass1)]
    {
        static map: u64 = 0;
    }
    #[cfg(not(rpass1))]
    {
        static MAP: u64 = 0;
    }
}
