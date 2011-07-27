// xfail-stage0
// error-pattern:meep
fn echo[T](c: chan[T], oc: chan[chan[T]]) {
    // Tests that the type argument in port gets
    // visited
    let p = port[T]();
    oc <| chan(p);

    let x;
    p |> x;
    c <| x;
}

fn main() { fail "meep"; }