#![warn(clippy::transmute_ptr_to_ptr)]
#![allow(clippy::borrow_as_ptr)]

// Make sure we can modify lifetimes, which is one of the recommended uses
// of transmute

// Make sure we can do static lifetime transmutes
unsafe fn transmute_lifetime_to_static<'a, T>(t: &'a T) -> &'static T {
    std::mem::transmute::<&'a T, &'static T>(t)
}

// Make sure we can do non-static lifetime transmutes
unsafe fn transmute_lifetime<'a, 'b, T>(t: &'a T, u: &'b T) -> &'b T {
    std::mem::transmute::<&'a T, &'b T>(t)
}

struct LifetimeParam<'a> {
    s: &'a str,
}

struct GenericParam<T> {
    t: T,
}

fn transmute_ptr_to_ptr() {
    let ptr = &1u32 as *const u32;
    let mut_ptr = &mut 1u32 as *mut u32;
    unsafe {
        // pointer-to-pointer transmutes; bad
        let _: *const f32 = std::mem::transmute(ptr);
        let _: *mut f32 = std::mem::transmute(mut_ptr);
        // ref-ref transmutes; bad
        let _: &f32 = std::mem::transmute(&1u32);
        let _: &f64 = std::mem::transmute(&1f32);
        // ^ this test is here because both f32 and f64 are the same TypeVariant, but they are not
        // the same type
        let _: &mut f32 = std::mem::transmute(&mut 1u32);
        let _: &GenericParam<f32> = std::mem::transmute(&GenericParam { t: 1u32 });
    }

    // these are recommendations for solving the above; if these lint we need to update
    // those suggestions
    let _ = ptr as *const f32;
    let _ = mut_ptr as *mut f32;
    let _ = unsafe { &*(&1u32 as *const u32 as *const f32) };
    let _ = unsafe { &mut *(&mut 1u32 as *mut u32 as *mut f32) };

    // transmute internal lifetimes, should not lint
    let s = "hello world".to_owned();
    let lp = LifetimeParam { s: &s };
    let _: &LifetimeParam<'static> = unsafe { std::mem::transmute(&lp) };
    let _: &GenericParam<&LifetimeParam<'static>> = unsafe { std::mem::transmute(&GenericParam { t: &lp }) };
}

// dereferencing raw pointers in const contexts, should not lint as it's unstable (issue 5959)
const _: &() = {
    struct Zst;
    let zst = &Zst;

    unsafe { std::mem::transmute::<&'static Zst, &'static ()>(zst) }
};

fn main() {}
