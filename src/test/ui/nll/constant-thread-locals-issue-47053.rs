// Regression test for issue #47053

#![feature(thread_local)]

#[thread_local]
static FOO: isize = 5;

fn main() {
    FOO = 6; //~ ERROR cannot assign to immutable static item `FOO` [E0594]
}
