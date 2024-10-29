// Ensure that we don't consider `const Trait` and `~const Trait` to
// match the macro fragment specifier `ty` as that would be a breaking
// change theoretically speaking. Syntactically trait object types can
// be "bare", i.e., lack the prefix `dyn`.
// By contrast, `?Trait` *does* match `ty` and therefore an arm like
// `?$Trait:path` would never be reached.
// See `parser/macro/mbe-bare-trait-object-maybe-trait-bound.rs`.

//@ check-pass

macro_rules! check {
    ($Type:ty) => { compile_error!("ty"); };
    (const $Trait:path) => {};
    (~const $Trait:path) => {};
}

check! { const Trait }
check! { ~const Trait }

fn main() {}
