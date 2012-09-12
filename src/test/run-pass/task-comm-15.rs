// xfail-win32
extern mod std;

fn start(c: pipes::Chan<int>, i0: int) {
    let mut i = i0;
    while i > 0 {
        c.send(0);
        i = i - 1;
    }
}

fn main() {
    // Spawn a task that sends us back messages. The parent task
    // is likely to terminate before the child completes, so from
    // the child's point of view the receiver may die. We should
    // drop messages on the floor in this case, and not crash!
    let (ch, p) = pipes::stream();
    task::spawn(|| start(ch, 10));
    p.recv();
}
