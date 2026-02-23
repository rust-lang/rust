//@ edition: 2024

// These nested macro expansions were found to cause span refinement to produce
// spans with a context that doesn't match the function body span, triggering
// a defensive check that discards the span.
//
// Reported in <https://github.com/rust-lang/rust/issues/147339>.

macro_rules! foo {
    ($($m:ident $($f:ident $v:tt)+),*) => {
        $($(macro_rules! $f { () => { $v } })+)*
        $(macro_rules! $m { () => { $(fn $f() -> i32 { $v })+ } })*
    }
}

foo!(m a 1 b 2, n c 3);
m!();
n!();

fn main() {}
