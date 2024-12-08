//@ run-rustfix

fn main() {
    2.e1; //~ERROR `{integer}` is a primitive type and therefore doesn't have fields
    2.E1; //~ERROR `{integer}` is a primitive type and therefore doesn't have fields
    2.f32; //~ERROR `{integer}` is a primitive type and therefore doesn't have fields
    2.f64; //~ERROR `{integer}` is a primitive type and therefore doesn't have fields
    2.e+12; //~ERROR `{integer}` is a primitive type and therefore doesn't have fields
    2.e-12; //~ERROR `{integer}` is a primitive type and therefore doesn't have fields
    2.e1f32; //~ERROR `{integer}` is a primitive type and therefore doesn't have fields
}
