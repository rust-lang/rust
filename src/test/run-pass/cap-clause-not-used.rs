// error-pattern: warning: Captured variable 'y' not used in closure
fn main() {
    let x = 5;
    let _y = sendfn[copy x]() { };
}
