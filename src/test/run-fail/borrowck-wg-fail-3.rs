// error-pattern:borrowed

fn main() {
    let x = @mut 3;
    let y: &mut int = x;
    *x = 5;
}
