// Issue #118164: recovery path leaving unemitted error behind
fn bar() -> String {
    #[cfg(feature = )]
    [1, 2, 3].iter().map().collect::<String>() //~ ERROR expected `;`, found `#`
    #[attr] //~ ERROR expected statement after outer attribute
}
fn main() {
    let _ = bar();
}
