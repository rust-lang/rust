// This should typecheck even though the type of e is not fully
// resolved when we finish typechecking the fn@.
fn main() {
    let e = @{mutable refs: [], n: 0};
    let f = fn@ () { log(error, e.n); };
    e.refs += [1];
}
