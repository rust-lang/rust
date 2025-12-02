----cargo
//~^ ERROR: unclosed frontmatter
//~| ERROR: frontmatters are experimental

//@ compile-flags: --crate-type lib

// Leading whitespace on the feature line prevents recovery. However
// the dashes quoted will not be used for recovery and the entire file
// should be treated as within the frontmatter block.

 #![feature(frontmatter)]

fn foo() -> &str {
    "----"
}
