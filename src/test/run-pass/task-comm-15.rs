// xfail-stage0

fn start(chan[int] c, int n) {
    let int i = n;

    while(i > 0) {
        c <| 0;
        i = i - 1;
    }
}

fn main() {
    let port[int] p = port();
    // Spawn a task that sends us back messages. The parent task
    // is likely to terminate before the child completes, so from
    // the child's point of view the receiver may die. We should
    // drop messages on the floor in this case, and not crash!
    auto child = spawn start(chan(p), 10);
    auto c; p |> c;
}
