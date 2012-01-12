// error-pattern:unresolved name: z
fn main() {
    let x = 5;
    let y = fn~[copy z, x]() {
    };
}