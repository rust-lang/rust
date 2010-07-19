io fn main() -> () {
   test00();
}

io fn test00() {
    let port[int] p = port();
    let chan[int] c = chan(p);
    c <| 42;
    let int r <- p;
}