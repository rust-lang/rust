// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir
// edition:2021
// build-pass

fn main() {
    let _ = async {
        let mut s = (String::new(),);
        s.0.push_str("abc");
        std::mem::drop(s);
        async {}.await;
    };
}
