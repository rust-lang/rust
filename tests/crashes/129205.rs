//@ known-bug: rust-lang/rust#129205

fn x<T: Copy>() {
    T::try_from();
}
