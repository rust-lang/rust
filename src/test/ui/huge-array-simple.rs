// FIXME (#23926): the error output is not consistent between a
// self-hosted and a cross-compiled setup. Skipping for now.

// ignore-test FIXME(#23926)

#![allow(exceeding_bitshifts)]

fn main() {
    let _fat : [u8; (1<<61)+(1<<31)] =
        [0; (1u64<<61) as usize +(1u64<<31) as usize];
}
