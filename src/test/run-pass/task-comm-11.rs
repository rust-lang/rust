// xfail-stage3

fn start(c: chan[chan[str]]) { let p: port[str] = port(); c <| chan(p); }

fn main() {
    let p: port[chan[str]] = port();
    let child = spawn start(chan(p));
    let c;
    p |> c;
}