// check-pass

#![feature(rustc_attrs)]
#![feature(negative_impls)]
#![feature(never_type)]

fn main() {
    enum Void {}

    #[rustc_auto_trait]
    trait Auto {}
    fn assert_auto<T: Auto>() {}
    assert_auto::<Void>();
    assert_auto::<!>();

    fn assert_send<T: Send>() {}
    assert_send::<Void>();
    assert_send::<!>();
}
