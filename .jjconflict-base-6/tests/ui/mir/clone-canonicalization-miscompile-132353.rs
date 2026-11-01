//! The mir-opt added in <https://github.com/rust-lang/rust/pull/128299> unfortunately seems to lead
//! to a miscompile (reported in <https://github.com/rust-lang/rust/issues/132353>, minimization
//! reproduced in this test file).
//@ revisions: release debug
// Note: it's not strictly cargo's release profile, but any non-zero opt-level was sufficient to
// reproduce the miscompile.
//@[release] compile-flags: -C opt-level=1
//@[debug] compile-flags: -C opt-level=0
//@ run-pass

fn pop_min(mut score2head: Vec<Option<usize>>) -> Option<usize> {
    loop {
        if let Some(col) = score2head[0] {
            score2head[0] = None;
            return Some(col);
        }
    }
}

fn main() {
    let min = pop_min(vec![Some(1)]);
    println!("min: {:?}", min);
    // panic happened on 1.83.0 beta in release mode but not debug mode.
    let _ = min.unwrap();
}
