fn force(f: fn()) { f(); }
fn main() {
    let x: int;
    force(fn&() {
        log(debug, x); //~ ERROR capture of possibly uninitialized variable: `x`
    });
}
