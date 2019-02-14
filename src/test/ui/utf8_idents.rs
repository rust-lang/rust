fn foo<
    'β, //~ ERROR non-ascii idents are not fully supported
    γ  //~ ERROR non-ascii idents are not fully supported
       //~^ WARN type parameter `γ` should have an upper camel case name
>() {}

struct X {
    δ: usize //~ ERROR non-ascii idents are not fully supported
}

pub fn main() {
    let α = 0.00001f64; //~ ERROR non-ascii idents are not fully supported
}
