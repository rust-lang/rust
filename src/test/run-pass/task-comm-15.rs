// xfail-stage1
// xfail-stage2
// xfail-stage3
// This test fails when run with multiple threads

fn start(c: chan[int], n: int) {
    let i: int = n;


    while i > 0 { c <| 0; i = i - 1; }
}

fn main() {
    let p: port[int] = port();
    // Spawn a task that sends us back messages. The parent task
    // is likely to terminate before the child completes, so from
    // the child's point of view the receiver may die. We should
    // drop messages on the floor in this case, and not crash!
    let child = spawn start(chan(p), 10);
    let c;
    p |> c;
}