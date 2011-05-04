// xfail-stage0
// xfail-stage1
// xfail-stage2
// xfail-stage1
// xfail-stage2
fn f() -> int {
    { ret 3; }
}

fn main() {
    assert(f() == 3);
}


