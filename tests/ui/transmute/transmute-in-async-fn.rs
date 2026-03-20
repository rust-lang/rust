//@compile-flags: -Zmir-enable-passes=+DataflowConstProp --crate-type lib
//@ edition:2021
pub async fn a() -> u32 {
    unsafe { std::mem::transmute(1u64) }
    //~^error: cannot transmute between types of different sizes, or dependently-sized types
}
