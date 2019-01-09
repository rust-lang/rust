// error-pattern:panic evaluated

#[allow(unused_variables)]
fn main() {
    let x = [panic!("panic evaluated"); 0];
}
