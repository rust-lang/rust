// xfail-stage0
// error-pattern:Spawn arguments with types containing parameters must be
fn main() {
// Similar to bind-parameterized-args
    fn echo[T](chan[T] c, chan[chan[T]] oc) {
        let port[T] p = port();
        oc <| chan(p);

        auto x;
        p |> x;
        c <| x;
    }

    auto p = port[int]();
    auto p2 = port[chan[int]]();

    spawn echo(chan(p), chan(p2));
}