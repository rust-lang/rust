//@ edition: 2024

// These nested macro expansions were found to cause span refinement to produce
// spans with a context that doesn't match the function body span, triggering
// a defensive check that discards the span.
//
// Reported in <https://github.com/rust-lang/rust/issues/147339>.

macro_rules! outer_macro {
    (
        $v:tt
    ) => {
        macro_rules! _other_macro_that_mentions_v {
            () => {
                $v
            };
        }
        macro_rules! inner_macro {
            () => {
                fn _function() -> i32 {
                    $v
                }
            };
        }
    };
}

outer_macro!(1);
inner_macro!();

fn main() {}
