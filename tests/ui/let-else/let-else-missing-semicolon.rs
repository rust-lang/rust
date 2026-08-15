fn main() {
    let Some(x) = Some(1) else {
        return;
    } //~ ERROR expected `;`, found keyword `let`
    let _ = "";
    let Some(x) = Some(1) else {
        panic!();
    } //~ ERROR expected `;`, found `}`
}
