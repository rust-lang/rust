// Ensure that we don't consider `const Trait` to match the macro fragment specifier `ty`
// as that would be a breaking change theoretically speaking.
//
// Syntactically trait object types can be "bare", i.e., lack the prefix `dyn`.
// By contrast, `?Trait` *does* match `ty` and therefore an arm like `?$Trait:path`
// would never be reached. See `parser/macro/macro-bare-trait-object-maybe-trait-bound.rs`.

//@ check-pass (KEEP THIS AS A PASSING TEST!)

macro_rules! check {
    ($ty:ty) => { compile_error!("ty"); }; // KEEP THIS RULE FIRST AND AS IS!

    // DON'T MODIFY THE MATCHERS BELOW UNLESS THE CONST TRAIT MODIFIER SYNTAX CHANGES!

    (const $Trait:path) => { /* KEEP THIS EMPTY! */ };
    // We don't need to check `[const] Trait` here since that matches the `ty` fragment
    // already anyway since `[` may begin a slice or array type. However, it'll then
    // subsequently fail due to #146122 (section 3).
}

check! { const Trait }

fn main() {}
