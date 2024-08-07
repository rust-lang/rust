#[target_feature(enable = "avx")] //~ ERROR attribute should be applied to a function definition
struct Avx {}

fn main() {}
