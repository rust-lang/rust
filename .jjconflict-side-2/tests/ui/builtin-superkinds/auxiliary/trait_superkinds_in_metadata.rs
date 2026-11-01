// Test library crate for cross-crate usages of traits inheriting
// from the builtin kinds. Mostly tests metadata correctness.

#![crate_type="lib"]

pub trait RequiresShare : Sync { }
pub trait RequiresRequiresShareAndSend : RequiresShare + Send { }
pub trait RequiresCopy : Copy { }
