type i32x4 = i32 ^ 4;
type f64x2 = f64 ^ 2;

fn test_int(e: i32) {
    let v = e as i32x4;
    v + v;
    v - v;
    v * v;
}

fn test_float(e: f64) {
    let v = e as f64x2;
    v + v;
    v - v;
    v * v;
}

fn main() {}
