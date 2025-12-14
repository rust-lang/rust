//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:hir-format-args-duplicated-captures.pp

const X: i32 = 42;

fn main() {
    let _ = format_args!("{X} {X}");
}
