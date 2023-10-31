// build-fail
// ignore-32bit

#![allow(arithmetic_overflow)]

fn main() {
    let _fat: [u8; (1<<61)+(1<<31)] = //~ ERROR too big for the current architecture
        [0; (1u64<<61) as usize +(1u64<<31) as usize];
}
