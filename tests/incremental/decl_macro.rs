//@ revisions: rpass1 rpass2

// issue#112680

#![feature(decl_macro)]

pub trait T {
    type Key;
    fn index_from_key(key: Self::Key) -> usize;
}

pub macro m($key_ty:ident, $val_ty:ident) {
    struct $key_ty {
        inner: usize,
    }

    impl T for $val_ty {
        type Key = $key_ty;

        fn index_from_key(key: Self::Key) -> usize {
            key.inner
        }
    }
}

m!(TestId, Test);

#[cfg(rpass1)]
struct Test(u32);

#[cfg(rpass2)]
struct Test;

fn main() {}
