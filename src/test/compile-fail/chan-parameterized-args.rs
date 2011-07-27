// xfail-stage0
// error-pattern:Spawn arguments with types containing parameters must be
fn main() {
    // Similar to bind-parameterized-args
    fn echo[T](c: chan[T], oc: chan[chan[T]]) {
        let p: port[T] = port();
        oc <| chan(p);

        let x;
        p |> x;
        c <| x;
    }

    let p = port[int]();
    let p2 = port[chan[int]]();

    spawn echo(chan(p), chan(p2));
}