// Issue 1974
// Don't double free the condition allocation
fn main() {
    let s = ~"hej";
    while s != ~"" {
        ret;
    }
}