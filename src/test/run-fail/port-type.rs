// xfail-stage0
// error-pattern:meep
fn echo[T](chan[T] c, chan[chan[T]] oc) {
  // Tests that the type argument in port gets
  // visited
        auto p = port[T]();
        oc <| chan(p);

        auto x;
        p |> x;
        c <| x;
}

fn main() {
  fail "meep";
}