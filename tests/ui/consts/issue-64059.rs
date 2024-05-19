//@ revisions: noopt opt opt_with_overflow_checks
//@[noopt]compile-flags: -C opt-level=0
//@[opt]compile-flags: -O
//@[opt_with_overflow_checks]compile-flags: -C overflow-checks=on -O

//@ run-pass

fn main() {
    let _ = -(-0.0);
}
