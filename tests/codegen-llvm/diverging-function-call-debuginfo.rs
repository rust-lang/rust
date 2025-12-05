/// Make sure that line debuginfo is correct for diverging calls under certain
/// conditions. In particular we want to ensure that the line number is never
/// 0, but we check the absence of 0 by looking for the expected exact line
/// numbers. Regression test for <https://github.com/rust-lang/rust/issues/59558>.

//@ compile-flags: -g -Clto -Copt-level=0
//@ no-prefer-dynamic

// First find the scope of both diverge() calls, namely this main() function.
// CHECK-DAG: [[MAIN_SCOPE:![0-9]+]] = distinct !DISubprogram(name: "main", linkageName: {{.*}}diverging_function_call_debuginfo{{.*}}main{{.*}}
fn main() {
    if True == False {
        // unreachable
        // Then find the DILocation with the correct line number for this call ...
        // CHECK-DAG: [[UNREACHABLE_CALL_DBG:![0-9]+]] = !DILocation(line: [[@LINE+1]], {{.*}}scope: [[MAIN_SCOPE]]
        diverge();
    }

    // ... and this call.
    // CHECK-DAG: [[LAST_CALL_DBG:![0-9]+]] = !DILocation(line: [[@LINE+1]], {{.*}}scope: [[MAIN_SCOPE]]
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

// Finally make sure both DILocations belong to each the respective diverge() call.
// CHECK-DAG: call void {{.*}}diverging_function_call_debuginfo{{.*}}diverge{{.*}} !dbg [[LAST_CALL_DBG]]
// CHECK-DAG: call void {{.*}}diverging_function_call_debuginfo{{.*}}diverge{{.*}} !dbg [[UNREACHABLE_CALL_DBG]]
