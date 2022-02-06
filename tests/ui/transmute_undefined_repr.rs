#![warn(clippy::transmute_undefined_repr)]
#![allow(clippy::unit_arg)]

fn value<T>() -> T {
    unimplemented!()
}

struct Empty;
struct Ty<T>(T);
struct Ty2<T, U>(T, U);

#[repr(C)]
struct Ty2C<T, U>(T, U);

fn main() {
    unsafe {
        let _: () = core::mem::transmute(value::<Empty>());
        let _: Empty = core::mem::transmute(value::<()>());

        let _: Ty<u32> = core::mem::transmute(value::<u32>());
        let _: Ty<u32> = core::mem::transmute(value::<u32>());

        let _: Ty2C<u32, i32> = core::mem::transmute(value::<Ty2<u32, i32>>()); // Lint, Ty2 is unordered
        let _: Ty2<u32, i32> = core::mem::transmute(value::<Ty2C<u32, i32>>()); // Lint, Ty2 is unordered

        let _: Ty2<u32, i32> = core::mem::transmute(value::<Ty<Ty2<u32, i32>>>()); // Ok, Ty2 types are the same
        let _: Ty<Ty2<u32, i32>> = core::mem::transmute(value::<Ty2<u32, i32>>()); // Ok, Ty2 types are the same

        let _: Ty2<u32, f32> = core::mem::transmute(value::<Ty<Ty2<u32, i32>>>()); // Lint, different Ty2 instances
        let _: Ty<Ty2<u32, i32>> = core::mem::transmute(value::<Ty2<u32, f32>>()); // Lint, different Ty2 instances

        let _: Ty<&()> = core::mem::transmute(value::<&()>());
        let _: &() = core::mem::transmute(value::<Ty<&()>>());

        let _: &Ty2<u32, f32> = core::mem::transmute(value::<Ty<&Ty2<u32, i32>>>()); // Lint, different Ty2 instances
        let _: Ty<&Ty2<u32, i32>> = core::mem::transmute(value::<&Ty2<u32, f32>>()); // Lint, different Ty2 instances

        let _: Ty<usize> = core::mem::transmute(value::<&Ty2<u32, i32>>()); // Ok, pointer to usize conversion
        let _: &Ty2<u32, i32> = core::mem::transmute(value::<Ty<usize>>()); // Ok, pointer to usize conversion

        let _: Ty<[u8; 8]> = core::mem::transmute(value::<Ty2<u32, i32>>()); // Ok, transmute to byte array
        let _: Ty2<u32, i32> = core::mem::transmute(value::<Ty<[u8; 8]>>()); // Ok, transmute from byte array
    }
}
