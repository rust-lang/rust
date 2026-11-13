// Sibling file referenced by `mod sub;` in edge_cases/mod.rs.
// Items here live in a non-mod.rs file → no fire even if they are definitions.
#[allow(dead_code)]
pub struct InSub;
#[allow(dead_code)]
pub fn in_sub() {}
