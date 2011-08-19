// This should typecheck even though the type of e is not fully
// resolved when we finish typechecking the lambda.
fn main() {
    let e = @{mutable refs: [], n: 0};
    let f = lambda () { log_err e.n; };
    e.refs += [1];
}
