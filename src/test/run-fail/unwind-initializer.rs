// error-pattern:fail

fn main() {
    let a: @int = {
        fail;
    };
}