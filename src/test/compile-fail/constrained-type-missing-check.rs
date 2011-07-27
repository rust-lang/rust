// -*- rust -*-
// xfail-stage0
// xfail-stage1
// xfail-stage2
// xfail-stage3
// error-pattern:Unsatisfied precondition

tag list { cons(int, @list); nil; }

type bubu = {x: int, y: int};

pred less_than(x: int, y: int) -> bool { ret x < y; }

type ordered_range = {low: int, high: int} : less_than(low, high);

fn main() {
    // Should fail to compile, b/c we're not doing the check
    // explicitly that a < b
    let a: int = 1;
    let b: int = 2;
    let c: ordered_range = {low: a, high: b};
    log c.low;
}