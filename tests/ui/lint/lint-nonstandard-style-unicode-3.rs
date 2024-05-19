#![allow(dead_code)]

#![forbid(non_upper_case_globals)]

// Some scripts (e.g., hiragana) don't have a concept of
// upper/lowercase

// 3. non_upper_case_globals

// Can only use non-lowercase letters.
// So this works:

static ラ: usize = 0;

// but this doesn't:

static τεχ: f32 = 3.14159265;
//~^ ERROR static variable `τεχ` should have an upper case name

// This has no limit at all on underscore usages.

static __密__封__线__内__禁__止__答__题__: bool = true;

fn main() {}
