// error-pattern:panic evaluated

#[allow(unused_variables)]
fn main() {
    // This used to trigger an LLVM assertion during compilation
    let x = [panic!("panic evaluated"); 2];
}
