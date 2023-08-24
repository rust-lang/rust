#![warn(clippy::large_const_arrays)]
#![allow(dead_code)]

#[derive(Clone, Copy)]
pub struct S {
    pub data: [u64; 32],
}

// Should lint
pub(crate) const FOO_PUB_CRATE: [u32; 1_000_000] = [0u32; 1_000_000];
pub const FOO_PUB: [u32; 1_000_000] = [0u32; 1_000_000];
const FOO: [u32; 1_000_000] = [0u32; 1_000_000];

// Good
pub(crate) const G_FOO_PUB_CRATE: [u32; 1_000] = [0u32; 1_000];
pub const G_FOO_PUB: [u32; 1_000] = [0u32; 1_000];
const G_FOO: [u32; 1_000] = [0u32; 1_000];

fn main() {
    // Should lint
    pub const BAR_PUB: [u32; 1_000_000] = [0u32; 1_000_000];
    const BAR: [u32; 1_000_000] = [0u32; 1_000_000];
    pub const BAR_STRUCT_PUB: [S; 5_000] = [S { data: [0; 32] }; 5_000];
    const BAR_STRUCT: [S; 5_000] = [S { data: [0; 32] }; 5_000];
    pub const BAR_S_PUB: [Option<&str>; 200_000] = [Some("str"); 200_000];
    const BAR_S: [Option<&str>; 200_000] = [Some("str"); 200_000];

    // Good
    pub const G_BAR_PUB: [u32; 1_000] = [0u32; 1_000];
    const G_BAR: [u32; 1_000] = [0u32; 1_000];
    pub const G_BAR_STRUCT_PUB: [S; 500] = [S { data: [0; 32] }; 500];
    const G_BAR_STRUCT: [S; 500] = [S { data: [0; 32] }; 500];
    pub const G_BAR_S_PUB: [Option<&str>; 200] = [Some("str"); 200];
    const G_BAR_S: [Option<&str>; 200] = [Some("str"); 200];
}
