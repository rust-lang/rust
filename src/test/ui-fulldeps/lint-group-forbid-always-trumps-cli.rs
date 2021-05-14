// aux-build:lint-group-plugin-test.rs
// compile-flags: -F unused -D forbidden_lint_groups -A unused -Z deduplicate-diagnostics=yes
//~^^ ERROR: allow(unused) incompatible
//~| WARNING: this was previously accepted
//~| ERROR: allow(unused) incompatible
//~| WARNING: this was previously accepted

fn main() {
    let x = 1;
}
