//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(non_local_definitions)]
#![feature(unsize, coerce_unsized)]

#[repr(packed)]
struct UnalignedPtr<'a, T: ?Sized>
    where T: 'a,
{
    data: &'a T,
}

fn main() {

    impl<'a, T, U> std::ops::CoerceUnsized<UnalignedPtr<'a, U>> for UnalignedPtr<'a, T>
        where
        T: std::marker::Unsize<U> + ?Sized,
        U: ?Sized,
    { }

    let arr = [1, 2, 3];
    let arr_unaligned: UnalignedPtr<[i32; 3]> = UnalignedPtr { data: &arr };
    let arr_unaligned: UnalignedPtr<[i32]> = arr_unaligned;
}
