//

fn foo<
    'β, //~ ERROR non-ascii idents are not fully supported
    γ  //~ ERROR non-ascii idents are not fully supported
>() {}

struct X {
    δ: usize //~ ERROR non-ascii idents are not fully supported
}

pub fn main() {
    let α = 0.00001f64; //~ ERROR non-ascii idents are not fully supported
}
