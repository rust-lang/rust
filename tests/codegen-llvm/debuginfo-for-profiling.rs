// Verify that additional discriminators are emitted for profiling with `-Zdebuginfo-for-profiling`:
//  - 2 discriminators are emitted without the flag in the test below
//  - 5 discriminators are emitted with the flag in the test below
//
//
//@ revisions: DEFAULT DEBUGINFO_FOR_PROFILING
//@ compile-flags: -Copt-level=2 -Cdebuginfo=line-tables-only
//@ [DEBUGINFO_FOR_PROFILING] compile-flags: -Zdebuginfo-for-profiling
// DEFAULT-COUNT-2: discriminator:
// DEFAULT-NOT: discriminator:
// DEBUGINFO_FOR_PROFILING-COUNT-5: discriminator:
// DEBUGINFO_FOR_PROFILING-NOT: discriminator:

fn main() {
    let mut sum = 0;

    for i in 1..=20 {
        if i % 2 == 0 {
            sum += compute(i);
        } else {
            sum += compute(i) * 2;
        }
    }

    println!("The total sum is: {}", sum);
}

fn compute(x: i32) -> i32 {
    if x < 10 { x * x } else { x + 5 }
}
