// xfail-test
// This checks that preemption works.

fn starve_main(alive: chan<int>) {
    #debug("signalling main");
    alive <| 1;
    #debug("starving main");
    let i: int = 0;
    while true { i += 1; }
}

fn main() {
    let alive: port<int> = port();
    #debug("main started");
    let s: task = spawn starve_main(chan(alive));
    let i: int;
    #debug("main waiting for alive signal");
    alive |> i;
    #debug("main got alive signal");
    while i < 50 { #debug("main iterated"); i += 1; }
    #debug("main completed");
}
