impure fn start(chan[chan[str]] c) {
    let port[str] p = port();
    c <| chan(p);
    auto a <- p;
    // auto b <- p; // Never read the second string.
}

impure fn main() {
    let port[chan[str]] p = port();
    auto child = spawn "start" start(chan(p));
    auto c <- p;
    c <| "A";
    c <| "B";
    yield;
}