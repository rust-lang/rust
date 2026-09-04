// https://github.com/rust-lang/rust/issues/16019

//@ check-pass

macro_rules! define_struct {
    ($rounds:expr) => (
        struct Struct {
            sk: [u32; $rounds + 1]
        }
    )
}

define_struct!(2);
