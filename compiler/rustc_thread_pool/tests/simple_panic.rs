use rustc_thred_pool::join;

#[test]
#[should_panic(expected = "should panic")]
fn simple_panic() {
    join(|| {}, || panic!("should panic"));
}
