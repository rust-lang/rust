// Test if ICE occurs when trying to intern an allocation that contains provenance to the global
// memory.

//@ build-pass
//@ compile-flags: -Zmir-enable-passes=+GVN

fn main() {
    let _x: Option<Box<[u8]>> = unsafe { std::mem::transmute((43_u8, &42_u8)) };
}
