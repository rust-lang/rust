//@ compile-flags: -Zdeduplicate-diagnostics=yes

#![warn(absolute_paths_not_starting_with_crate, reason = 0)]
//~^ ERROR malformed
//~| NOTE expected a string literal here
//~| NOTE for more information, visit <https://doc.rust-lang.org/reference/attributes/diagnostics.html#lint-check-attributes>
#![warn(anonymous_parameters, reason = b"consider these, for we have condemned them")]
//~^ ERROR malformed
//~| NOTE expected a normal string literal, not a byte string literal
#![warn(bare_trait_objects, reasons = "leaders to no sure land, guides their bearings lost")]
//~^ ERROR malformed
//~| NOTE the only valid argument here is `reason`
//~| NOTE for more information, visit <https://doc.rust-lang.org/reference/attributes/diagnostics.html#lint-check-attributes>
#![warn(unsafe_code, blerp = "or in league with robbers have reversed the signposts")]
//~^ ERROR malformed
//~| NOTE the only valid argument here is `reason`
//~| NOTE for more information, visit <https://doc.rust-lang.org/reference/attributes/diagnostics.html#lint-check-attributes>
#![warn(elided_lifetimes_in_paths, reason("disrespectful to ancestors", "irresponsible to heirs"))]
//~^ ERROR malformed
//~| NOTE didn't expect any arguments here
//~| NOTE for more information, visit <https://doc.rust-lang.org/reference/attributes/diagnostics.html#lint-check-attributes>
#![warn(ellipsis_inclusive_range_patterns, reason = "born barren", reason = "a freak growth")]
//~^ ERROR malformed
//~| NOTE expected reason = "..." to be the last argument
//~| NOTE for more information, visit <https://doc.rust-lang.org/reference/attributes/diagnostics.html#lint-check-attributes>
#![warn(keyword_idents, reason = "root in rubble", macro_use_extern_crate)]
//~^ ERROR malformed
//~| NOTE expected reason = "..." to be the last argument
//~| NOTE for more information, visit <https://doc.rust-lang.org/reference/attributes/diagnostics.html#lint-check-attributes>
#![warn(missing_copy_implementations, reason)]
//~^ WARN unknown lint
//~| NOTE `#[warn(unknown_lints)]` on by default

fn main() {}
