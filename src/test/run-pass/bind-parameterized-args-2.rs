fn main() {
    fn echo<T>(c: int, x: fn(&T)) { log_err "wee"; }

    let y = bind echo(42, _);

    y(fn (i: &str) { });
}
