// ignore-tidy-linelength
// compile-flags:-Zprint-mono-items=eager
// compile-flags:-Zinline-in-all-cgus

#![deny(dead_code)]
#![feature(start)]

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
    B(T2)
}

impl<T1, T2> Drop for EnumWithDrop<T1, T2> {
    fn drop(&mut self) {}
}

enum EnumNoDrop<T1, T2> {
    A(T1),
    B(T2)
}


struct NonGenericNoDrop(i32);

struct NonGenericWithDrop(i32);
//~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<generic_drop_glue::NonGenericWithDrop[0]> @@ generic_drop_glue-cgu.0[Internal]

impl Drop for NonGenericWithDrop {
    //~ MONO_ITEM fn generic_drop_glue::{{impl}}[2]::drop[0]
    fn drop(&mut self) {}
}

//~ MONO_ITEM fn generic_drop_glue::start[0]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    //~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<generic_drop_glue::StructWithDrop[0]<i8, char>> @@ generic_drop_glue-cgu.0[Internal]
    //~ MONO_ITEM fn generic_drop_glue::{{impl}}[0]::drop[0]<i8, char>
    let _ = StructWithDrop { x: 0i8, y: 'a' }.x;

    //~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<generic_drop_glue::StructWithDrop[0]<&str, generic_drop_glue::NonGenericNoDrop[0]>> @@ generic_drop_glue-cgu.0[Internal]
    //~ MONO_ITEM fn generic_drop_glue::{{impl}}[0]::drop[0]<&str, generic_drop_glue::NonGenericNoDrop[0]>
    let _ = StructWithDrop { x: "&str", y: NonGenericNoDrop(0) }.y;

    // Should produce no drop glue
    let _ = StructNoDrop { x: 'a', y: 0u32 }.x;

    // This is supposed to generate drop-glue because it contains a field that
    // needs to be dropped.
    //~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<generic_drop_glue::StructNoDrop[0]<generic_drop_glue::NonGenericWithDrop[0], f64>> @@ generic_drop_glue-cgu.0[Internal]
    let _ = StructNoDrop { x: NonGenericWithDrop(0), y: 0f64 }.y;

    //~ MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<generic_drop_glue::EnumWithDrop[0]<i32, i64>> @@ generic_drop_glue-cgu.0[Internal]
    //~ MONO_ITEM fn generic_drop_glue::{{impl}}[1]::drop[0]<i32, i64>
    let _ = match EnumWithDrop::A::<i32, i64>(0) {
        EnumWithDrop::A(x) => x,
        EnumWithDrop::B(x) => x as i32
    };

    //~MONO_ITEM fn core::ptr[0]::real_drop_in_place[0]<generic_drop_glue::EnumWithDrop[0]<f64, f32>> @@ generic_drop_glue-cgu.0[Internal]
    //~ MONO_ITEM fn generic_drop_glue::{{impl}}[1]::drop[0]<f64, f32>
    let _ = match EnumWithDrop::B::<f64, f32>(1.0) {
        EnumWithDrop::A(x) => x,
        EnumWithDrop::B(x) => x as f64
    };

    let _ = match EnumNoDrop::A::<i32, i64>(0) {
        EnumNoDrop::A(x) => x,
        EnumNoDrop::B(x) => x as i32
    };

    let _ = match EnumNoDrop::B::<f64, f32>(1.0) {
        EnumNoDrop::A(x) => x,
        EnumNoDrop::B(x) => x as f64
    };

    0
}
