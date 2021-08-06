// check-pass

#![feature(explicit_generic_args_with_impl_trait)]

fn f<T: ?Sized>(_: impl AsRef<T>, _: impl AsRef<T>) {}

fn main() {
    f::<[u8]>("a", b"a");
}
