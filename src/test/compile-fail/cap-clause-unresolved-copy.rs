// error-pattern:unresolved name
fn main() {
    let x = 5;
    let y = fn~(copy z, copy x) {
    };
}
