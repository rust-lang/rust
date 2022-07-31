use super::lgammaf_r;

#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn lgammaf(x: f32) -> f32 {
    lgammaf_r(x).0
}
