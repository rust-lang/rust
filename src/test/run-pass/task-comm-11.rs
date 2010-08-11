io fn start(chan[chan[str]] c) {
    let port[str] p = port();
    c <| chan(p);
}

io fn main() {
    let port[chan[str]] p = port();
    auto child = spawn "child" start(chan(p));
    auto c <- p;
}