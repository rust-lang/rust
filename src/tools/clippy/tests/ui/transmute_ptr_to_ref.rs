#![warn(clippy::transmute_ptr_to_ref)]
#![allow(
    clippy::match_single_binding,
    clippy::unnecessary_cast,
    clippy::missing_transmute_annotations
)]

fn ptr_to_ref<T, U>(p: *const T, m: *mut T, o: *const U, om: *mut U) {
    unsafe {
        let _: &T = std::mem::transmute(p);
        //~^ transmute_ptr_to_ref
        let _: &T = &*p;

        let _: &mut T = std::mem::transmute(m);
        //~^ transmute_ptr_to_ref
        let _: &mut T = &mut *m;

        let _: &T = std::mem::transmute(m);
        //~^ transmute_ptr_to_ref
        let _: &T = &*m;

        let _: &mut T = std::mem::transmute(p as *mut T);
        //~^ transmute_ptr_to_ref
        let _ = &mut *(p as *mut T);

        let _: &T = std::mem::transmute(o);
        //~^ transmute_ptr_to_ref
        let _: &T = &*(o as *const T);

        let _: &mut T = std::mem::transmute(om);
        //~^ transmute_ptr_to_ref
        let _: &mut T = &mut *(om as *mut T);

        let _: &T = std::mem::transmute(om);
        //~^ transmute_ptr_to_ref
        let _: &T = &*(om as *const T);
    }
}

fn issue1231() {
    struct Foo<'a, T> {
        bar: &'a T,
    }

    let raw = 42 as *const i32;
    let _: &Foo<u8> = unsafe { std::mem::transmute::<_, &Foo<_>>(raw) };
    //~^ transmute_ptr_to_ref

    let _: &Foo<&u8> = unsafe { std::mem::transmute::<_, &Foo<&_>>(raw) };
    //~^ transmute_ptr_to_ref

    type Bar<'a> = &'a u8;
    let raw = 42 as *const i32;
    unsafe { std::mem::transmute::<_, Bar>(raw) };
    //~^ transmute_ptr_to_ref
}

#[derive(Clone, Copy)]
struct PtrRefNamed<'a> {
    ptr: *const &'a u32,
}
#[derive(Clone, Copy)]
struct PtrRef<'a>(*const &'a u32);
#[derive(Clone, Copy)]
struct PtrSliceRef<'a>(*const [&'a str]);
#[derive(Clone, Copy)]
struct PtrSlice(*const [i32]);
#[derive(Clone, Copy)]
struct Ptr(*const u32);
impl std::ops::Add for Ptr {
    type Output = Self;
    fn add(self, _: Self) -> Self {
        self
    }
}
mod ptr_mod {
    #[derive(Clone, Copy)]
    pub struct Ptr(*const u32);
}
fn issue1966(u: PtrSlice, v: PtrSliceRef, w: Ptr, x: PtrRefNamed, y: PtrRef, z: ptr_mod::Ptr) {
    unsafe {
        let _: &i32 = std::mem::transmute(w);
        //~^ transmute_ptr_to_ref
        let _: &u32 = std::mem::transmute(w);
        //~^ transmute_ptr_to_ref
        let _: &&u32 = core::mem::transmute(x);
        //~^ transmute_ptr_to_ref
        // The field is not accessible. The program should not generate code
        // that accesses the field.
        let _: &u32 = std::mem::transmute(z);
        let _ = std::mem::transmute::<_, &u32>(w);
        //~^ transmute_ptr_to_ref
        let _: &[&str] = core::mem::transmute(v);
        //~^ transmute_ptr_to_ref
        let _ = std::mem::transmute::<_, &[i32]>(u);
        //~^ transmute_ptr_to_ref
        let _: &&u32 = std::mem::transmute(y);
        //~^ transmute_ptr_to_ref
        let _: &u32 = std::mem::transmute(w + w);
        //~^ transmute_ptr_to_ref
    }
}

fn issue8924<'a, 'b, 'c>(x: *const &'a u32, y: *const &'b u32) -> &'c &'b u32 {
    unsafe {
        match 0 {
            0 => std::mem::transmute(x),
            //~^ transmute_ptr_to_ref
            1 => std::mem::transmute(y),
            //~^ transmute_ptr_to_ref
            2 => std::mem::transmute::<_, &&'b u32>(x),
            //~^ transmute_ptr_to_ref
            _ => std::mem::transmute::<_, &&'b u32>(y),
            //~^ transmute_ptr_to_ref
        }
    }
}

#[clippy::msrv = "1.38"]
fn meets_msrv<'a, 'b, 'c>(x: *const &'a u32) -> &'c &'b u32 {
    unsafe {
        let a = 0u32;
        let a = &a as *const u32;
        let _: &u32 = std::mem::transmute(a);
        //~^ transmute_ptr_to_ref
        let _: &u32 = std::mem::transmute::<_, &u32>(a);
        //~^ transmute_ptr_to_ref
        match 0 {
            0 => std::mem::transmute(x),
            //~^ transmute_ptr_to_ref
            _ => std::mem::transmute::<_, &&'b u32>(x),
            //~^ transmute_ptr_to_ref
        }
    }
}

#[clippy::msrv = "1.37"]
fn under_msrv<'a, 'b, 'c>(x: *const &'a u32, y: PtrRef) -> &'c &'b u32 {
    unsafe {
        let a = 0u32;
        let a = &a as *const u32;
        let _: &u32 = std::mem::transmute(a);
        //~^ transmute_ptr_to_ref
        let _: &u32 = std::mem::transmute::<_, &u32>(a);
        //~^ transmute_ptr_to_ref
        let _ = std::mem::transmute::<_, &u32>(Ptr(a));
        //~^ transmute_ptr_to_ref
        match 0 {
            0 => std::mem::transmute(x),
            //~^ transmute_ptr_to_ref
            1 => std::mem::transmute::<_, &&'b u32>(x),
            //~^ transmute_ptr_to_ref
            2 => std::mem::transmute(y),
            //~^ transmute_ptr_to_ref
            _ => std::mem::transmute::<_, &&'b u32>(y),
            //~^ transmute_ptr_to_ref
        }
    }
}

// handle DSTs
fn issue13357(ptr: *const [i32], s_ptr: *const &str, a_s_ptr: *const [&str]) {
    unsafe {
        // different types, without erased regions
        let _ = core::mem::transmute::<_, &[u32]>(ptr);
        //~^ transmute_ptr_to_ref
        let _: &[u32] = core::mem::transmute(ptr);
        //~^ transmute_ptr_to_ref

        // different types, with erased regions
        let _ = core::mem::transmute::<_, &[&[u8]]>(a_s_ptr);
        //~^ transmute_ptr_to_ref
        let _: &[&[u8]] = core::mem::transmute(a_s_ptr);
        //~^ transmute_ptr_to_ref

        // same type, without erased regions
        let _ = core::mem::transmute::<_, &[i32]>(ptr);
        //~^ transmute_ptr_to_ref
        let _: &[i32] = core::mem::transmute(ptr);
        //~^ transmute_ptr_to_ref

        // same type, with erased regions
        let _ = core::mem::transmute::<_, &[&str]>(a_s_ptr);
        //~^ transmute_ptr_to_ref
        let _: &[&str] = core::mem::transmute(a_s_ptr);
        //~^ transmute_ptr_to_ref
    }
}

fn main() {}
