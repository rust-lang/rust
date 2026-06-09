#![warn(clippy::large_const_arrays)]
#![allow(dead_code)]

#[derive(Clone, Copy)]
pub struct S {
    pub data: [u64; 32],
}

// Should lint
pub(crate) const FOO_PUB_CRATE: [u32; 1_000_000] = [0u32; 1_000_000];
//~^ large_const_arrays
pub const FOO_PUB: [u32; 1_000_000] = [0u32; 1_000_000];
//~^ large_const_arrays
const FOO_COMPUTED: [u32; 1_000 * 100] = [0u32; 1_000 * 100];
//~^ large_const_arrays
const FOO: [u32; 1_000_000] = [0u32; 1_000_000];
//~^ large_const_arrays

// Good
pub(crate) const G_FOO_PUB_CRATE: [u32; 250] = [0u32; 250];
pub const G_FOO_PUB: [u32; 250] = [0u32; 250];
const G_FOO_COMPUTED: [u32; 25 * 10] = [0u32; 25 * 10];
const G_FOO: [u32; 250] = [0u32; 250];

fn main() {
    // Should lint
    pub const BAR_PUB: [u32; 1_000_000] = [0u32; 1_000_000];
    //~^ large_const_arrays
    const BAR: [u32; 1_000_000] = [0u32; 1_000_000];
    //~^ large_const_arrays
    pub const BAR_STRUCT_PUB: [S; 5_000] = [S { data: [0; 32] }; 5_000];
    //~^ large_const_arrays
    const BAR_STRUCT: [S; 5_000] = [S { data: [0; 32] }; 5_000];
    //~^ large_const_arrays
    pub const BAR_S_PUB: [Option<&str>; 200_000] = [Some("str"); 200_000];
    //~^ large_const_arrays
    const BAR_S: [Option<&str>; 200_000] = [Some("str"); 200_000];
    //~^ large_const_arrays

    // Good
    pub const G_BAR_PUB: [u32; 250] = [0u32; 250];
    const G_BAR: [u32; 250] = [0u32; 250];
    pub const G_BAR_STRUCT_PUB: [S; 4] = [S { data: [0; 32] }; 4];
    const G_BAR_STRUCT: [S; 4] = [S { data: [0; 32] }; 4];
    pub const G_BAR_S_PUB: [Option<&str>; 50] = [Some("str"); 50];
    const G_BAR_S: [Option<&str>; 50] = [Some("str"); 50];
}
