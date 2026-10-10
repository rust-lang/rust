//@ compile-flags: -Zmir-enable-passes=+DataflowConstProp --crate-type lib
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ [next] compile-flags: -Znext-solver
//@ edition:2021
pub async fn a() -> u32 {
    unsafe { std::mem::transmute(1u64) }
    //~^error: cannot transmute between types of different sizes, or dependently-sized types
}

pub async fn b() -> u32 {
    let closure = || unsafe { std::mem::transmute(1u64) };
    //~^ ERROR: cannot transmute between types of different sizes, or dependently-sized types [E0512]
    closure()
}
