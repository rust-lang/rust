fn dup(x: ~int) -> ~(~int,~int) { ~(x, x) } //~ ERROR use of moved variable
fn main() {
    dup(~3);
}
