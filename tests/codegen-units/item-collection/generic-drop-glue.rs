//@ compile-flags:-Clink-dead-code
//@ compile-flags: -O

#![deny(dead_code)]
#![crate_type = "lib"]

struct StructWithDrop<T1, T2> {
    x: T1,
    y: T2,
}

impl<T1, T2> Drop for StructWithDrop<T1, T2> {
    fn drop(&mut self) {}
}

struct StructNoDrop<T1, T2> {
    x: T1,
    y: T2,
}

enum EnumWithDrop<T1, T2> {
    A(T1),
    B(T2),
}

impl<T1, T2> Drop for EnumWithDrop<T1, T2> {
    fn drop(&mut self) {}
}

enum EnumNoDrop<T1, T2> {
    A(T1),
    B(T2),
}

struct NonGenericNoDrop(#[allow(dead_code)] i32);

struct NonGenericWithDrop(#[allow(dead_code)] i32);
//~ MONO_ITEM fn std::ptr::drop_in_place::<NonGenericWithDrop> - shim(Some(NonGenericWithDrop)) @@ generic_drop_glue-cgu.0[Internal]

impl Drop for NonGenericWithDrop {
    //~ MONO_ITEM fn <NonGenericWithDrop as std::ops::Drop>::drop
    fn drop(&mut self) {}
}

//~ MONO_ITEM fn start
#[no_mangle]
pub fn start(_: isize, _: *const *const u8) -> isize {
    //~ MONO_ITEM fn std::ptr::drop_in_place::<StructWithDrop<i8, char>> - shim(Some(StructWithDrop<i8, char>)) @@ generic_drop_glue-cgu.0[Internal]
    //~ MONO_ITEM fn <StructWithDrop<i8, char> as std::ops::Drop>::drop
    let _ = StructWithDrop { x: 0i8, y: 'a' }.x;

    //~ MONO_ITEM fn std::ptr::drop_in_place::<StructWithDrop<&str, NonGenericNoDrop>> - shim(Some(StructWithDrop<&str, NonGenericNoDrop>)) @@ generic_drop_glue-cgu.0[Internal]
    //~ MONO_ITEM fn <StructWithDrop<&str, NonGenericNoDrop> as std::ops::Drop>::drop
    let _ = StructWithDrop { x: "&str", y: NonGenericNoDrop(0) }.y;

    // Should produce no drop glue
    let _ = StructNoDrop { x: 'a', y: 0u32 }.x;

    // This is supposed to generate drop-glue because it contains a field that
    // needs to be dropped.
    //~ MONO_ITEM fn std::ptr::drop_in_place::<StructNoDrop<NonGenericWithDrop, f64>> - shim(Some(StructNoDrop<NonGenericWithDrop, f64>)) @@ generic_drop_glue-cgu.0[Internal]
    let _ = StructNoDrop { x: NonGenericWithDrop(0), y: 0f64 }.y;

    //~ MONO_ITEM fn std::ptr::drop_in_place::<EnumWithDrop<i32, i64>> - shim(Some(EnumWithDrop<i32, i64>)) @@ generic_drop_glue-cgu.0[Internal]
    //~ MONO_ITEM fn <EnumWithDrop<i32, i64> as std::ops::Drop>::drop
    let _ = match EnumWithDrop::A::<i32, i64>(0) {
        EnumWithDrop::A(x) => x,
        EnumWithDrop::B(x) => x as i32,
    };

    //~ MONO_ITEM fn std::ptr::drop_in_place::<EnumWithDrop<f64, f32>> - shim(Some(EnumWithDrop<f64, f32>)) @@ generic_drop_glue-cgu.0[Internal]
    //~ MONO_ITEM fn <EnumWithDrop<f64, f32> as std::ops::Drop>::drop
    let _ = match EnumWithDrop::B::<f64, f32>(1.0) {
        EnumWithDrop::A(x) => x,
        EnumWithDrop::B(x) => x as f64,
    };

    let _ = match EnumNoDrop::A::<i32, i64>(0) {
        EnumNoDrop::A(x) => x,
        EnumNoDrop::B(x) => x as i32,
    };

    let _ = match EnumNoDrop::B::<f64, f32>(1.0) {
        EnumNoDrop::A(x) => x,
        EnumNoDrop::B(x) => x as f64,
    };

    0
}
