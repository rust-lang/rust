// build-pass

macro_rules! perftools_inline {
    ($($item:tt)*) => (
        $($item)*
    );
}
mod state {
    pub struct RawFloatState;
    impl RawFloatState {
        perftools_inline! {
            pub(super) fn new() {}
        }
    }
}
