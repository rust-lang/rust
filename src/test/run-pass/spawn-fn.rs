// xfail-stage1
// xfail-stage2
// xfail-stage3
// -*- rust -*-

fn x(s: str, n: int) { log s; log n; }

fn main() {
    spawn x("hello from first spawned fn", 65);
    spawn x("hello from second spawned fn", 66);
    spawn x("hello from third spawned fn", 67);
    let i: int = 30;
    while i > 0 { i = i - 1; log "parent sleeping"; yield; }
}