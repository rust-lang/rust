/// Make sure that line debuginfo is correct for diverging calls under certain
/// conditions. In particular we want to ensure that the line number is never
/// 0, but we check the absence of 0 by looking for the expected exact line
/// numbers. Regression test for <https://github.com/rust-lang/rust/issues/59558>.

//@ compile-flags: -g -Clto -Copt-level=0
//@ no-prefer-dynamic

fn main() {
    if True == False {
        // unreachable
        // CHECK-DAG: [[UNREACHABLE_CALL_DBG:![0-9]+]] = !DILocation(line: [[@LINE+1]], column: 9, scope:
        diverge();
    }

    // CHECK-DAG: [[LAST_CALL_DBG:![0-9]+]] = !DILocation(line: [[@LINE+1]], column: 5, scope:
    diverge();
}

#[derive(PartialEq)]
pub enum MyBool {
    True,
    False,
}

use MyBool::*;

fn diverge() -> ! {
    panic!();
}

// CHECK-DAG: call void {{.*}}diverging_function_call_debuginfo{{.*}}diverge{{.*}} !dbg [[LAST_CALL_DBG]]
// CHECK-DAG: call void {{.*}}diverging_function_call_debuginfo{{.*}}diverge{{.*}} !dbg [[UNREACHABLE_CALL_DBG]]
