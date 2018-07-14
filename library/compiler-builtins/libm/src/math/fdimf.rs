use super::isnanf;

pub fn fdimf(x: f32, y: f32) -> f32 {
    if isnanf(x) {
        x
    } else if isnanf(y) {
        y
    } else {
        if x > y {
            x - y
        } else {
            0.0
        }
    }
}
