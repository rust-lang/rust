//@ check-pass

// Issue: https://github.com/rust-lang/rust/issues/73747
// ICE that seems to happen in type normalization when dealing with casts

#[repr(transparent)]
struct NonNullRawComPtr<T: ComInterface> {
    inner: std::ptr::NonNull<<T as ComInterface>::VTable>,
}

trait ComInterface {
    type VTable;
}

extern "C" fn invoke<T: ComInterface>(_: Option<NonNullRawComPtr<T>>) {}

fn main() {}
