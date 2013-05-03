#[simd]
struct vec4<T>(T, T, T, T); //~ ERROR SIMD vector cannot be generic

#[simd]
struct empty; //~ ERROR SIMD vector cannot be empty

#[simd]
struct i64f64(i64, f64); //~ ERROR SIMD vector should be homogeneous

#[simd]
struct int4(int, int, int, int); //~ ERROR SIMD vector element type should be machine type

fn main() {}
