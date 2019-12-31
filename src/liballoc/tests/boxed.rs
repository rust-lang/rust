use std::mem::MaybeUninit;
use std::ptr::NonNull;

#[test]
fn unitialized_zero_size_box() {
    assert_eq!(
        &*Box::<()>::new_uninit() as *const _,
        NonNull::<MaybeUninit<()>>::dangling().as_ptr(),
    );
    assert_eq!(
        Box::<[()]>::new_uninit_slice(4).as_ptr(),
        NonNull::<MaybeUninit<()>>::dangling().as_ptr(),
    );
    assert_eq!(
        Box::<[String]>::new_uninit_slice(0).as_ptr(),
        NonNull::<MaybeUninit<String>>::dangling().as_ptr(),
    );
}
