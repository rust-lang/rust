#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn scalbn(x: f64, n: i32) -> f64 {
    super::generic::scalbn(x, n)
}
