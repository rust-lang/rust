// compile-flags: -Z deduplicate-diagnostics=yes -F unused -D forbidden_lint_groups -A unused
//~^ ERROR: allow(unused) incompatible
//~| WARNING: this was previously accepted
//~| ERROR: allow(unused) incompatible
//~| WARNING: this was previously accepted

fn main() {}
