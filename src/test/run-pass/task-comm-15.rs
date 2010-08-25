io fn start(chan[int] c, int n) {
    let int i = n;

    while(i > 0) {
        c <| 0;
        i = i - 1;
    }
}

io fn main() {
    let port[int] p = port();
    auto child = spawn thread "child" start(chan(p), 10);
    auto c <- p;
}