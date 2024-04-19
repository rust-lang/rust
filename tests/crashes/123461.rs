//@ known-bug: #123461

fn main() {
    let _: [_; unsafe { std::mem::transmute(|o_b: Option<_>| {}) }];
}
