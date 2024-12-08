//@ check-pass

#[repr(transparent)]
struct NonNullRawComPtr<T: ComInterface> {
    inner: std::ptr::NonNull<<T as ComInterface>::VTable>,
}

trait ComInterface {
    type VTable;
}

extern "C" fn invoke<T: ComInterface>(_: Option<NonNullRawComPtr<T>>) {}

fn main() {}
