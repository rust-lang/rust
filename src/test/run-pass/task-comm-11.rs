// xfail-stage0
// xfail-stage1
// xfail-stage2
fn start(chan[chan[str]] c) {
    let port[str] p = port();
    c <| chan(p);
}

fn main() {
    let port[chan[str]] p = port();
    auto child = spawn "child" start(chan(p));
    auto c; c <- p;
}
