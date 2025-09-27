//@ run-fail
//@ compile-flags: -Cstrip=none -Cdebuginfo=line-tables-only -Copt-level=0
//@ exec-env:RUST_BACKTRACE=1
//@ regex-error-pattern: location-detail-unwrap-multiline\.rs:10(:10)?:\n
//@ needs-unwind

fn main() {
    let opt: Option<u32> = None;
    opt
        .unwrap();
}
