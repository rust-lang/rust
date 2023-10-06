// check-fail

#![feature(ptr_from_ref)]

extern "C" {
    // N.B., mutability can be easily incorrect in FFI calls -- as
    // in C, the default is mutable pointers.
    fn ffi(c: *mut u8);
    fn int_ffi(c: *mut i32);
}

fn static_u8() -> &'static u8 {
    &8
}

unsafe fn ref_to_mut() {
    let num = &3i32;

    let _num = &mut *(num as *const i32 as *mut i32);
    //~^ ERROR casting `&T` to `&mut T` is undefined behavior
    let _num = &mut *(num as *const i32).cast_mut();
    //~^ ERROR casting `&T` to `&mut T` is undefined behavior
    let _num = &mut *std::ptr::from_ref(num).cast_mut();
    //~^ ERROR casting `&T` to `&mut T` is undefined behavior
    let _num = &mut *std::ptr::from_ref({ num }).cast_mut();
    //~^ ERROR casting `&T` to `&mut T` is undefined behavior
    let _num = &mut *{ std::ptr::from_ref(num) }.cast_mut();
    //~^ ERROR casting `&T` to `&mut T` is undefined behavior
    let _num = &mut *(std::ptr::from_ref({ num }) as *mut i32);
    //~^ ERROR casting `&T` to `&mut T` is undefined behavior
    let _num = &mut *(num as *const i32).cast::<i32>().cast_mut();
    //~^ ERROR casting `&T` to `&mut T` is undefined behavior
    let _num = &mut *(num as *const i32).cast::<i32>().cast_mut().cast_const().cast_mut();
    //~^ ERROR casting `&T` to `&mut T` is undefined behavior
    let _num = &mut *(std::ptr::from_ref(static_u8()) as *mut i32);
    //~^ ERROR casting `&T` to `&mut T` is undefined behavior
    let _num = &mut *std::mem::transmute::<_, *mut i32>(num);
    //~^ ERROR casting `&T` to `&mut T` is undefined behavior
    let _num = &mut *(std::mem::transmute::<_, *mut i32>(num) as *mut i32);
    //~^ ERROR casting `&T` to `&mut T` is undefined behavior
    let _num = &mut *std::cell::UnsafeCell::raw_get(
    //~^ ERROR casting `&T` to `&mut T` is undefined behavior
        num as *const i32 as *const std::cell::UnsafeCell<i32>
    );

    let deferred = num as *const i32 as *mut i32;
    let _num = &mut *deferred;
    //~^ ERROR casting `&T` to `&mut T` is undefined behavior
    let deferred = (std::ptr::from_ref(num) as *const i32 as *const i32).cast_mut() as *mut i32;
    let _num = &mut *deferred;
    //~^ ERROR casting `&T` to `&mut T` is undefined behavior
    let deferred_rebind = deferred;
    let _num = &mut *deferred_rebind;
    //~^ ERROR casting `&T` to `&mut T` is undefined behavior
    let _num = &mut *(num as *const _ as usize as *mut i32);
    //~^ ERROR casting `&T` to `&mut T` is undefined behavior
    let _num = &mut *(std::mem::transmute::<_, *mut _>(num as *const i32) as *mut i32);
    //~^ ERROR casting `&T` to `&mut T` is undefined behavior

    static NUM: &'static i32 = &2;
    let num = NUM as *const i32 as *mut i32;
    let num = num;
    let num = num;
    let _num = &mut *num;
    //~^ ERROR casting `&T` to `&mut T` is undefined behavior

    let cell = &std::cell::UnsafeCell::new(0);
    let _num = &mut *(cell as *const _ as *mut i32);
    //~^ ERROR casting `&T` to `&mut T` is undefined behavior

    unsafe fn generic_ref_cast_mut<T>(this: &T) -> &mut T {
        &mut *((this as *const _) as *mut _)
        //~^ ERROR casting `&T` to `&mut T` is undefined behavior
    }

    fn as_mut<T>(x: &T) -> &mut T {
        unsafe { &mut *std::cell::UnsafeCell::raw_get(x as *const _ as *const _) }
        //~^ ERROR casting `&T` to `&mut T` is undefined behavior
    }

    fn as_mut_i32(x: &i32) -> &mut i32 {
        unsafe { &mut *std::cell::UnsafeCell::raw_get(x as *const _ as *const _) }
        //~^ ERROR casting `&T` to `&mut T` is undefined behavior
    }
}

unsafe fn assign_to_ref() {
    let s = String::from("Hello");
    let a = &s;
    let num = &3i32;

    *(a as *const _ as *mut _) = String::from("Replaced");
    //~^ ERROR assigning to `&T` is undefined behavior
    *(a as *const _ as *mut String) += " world";
    //~^ ERROR assigning to `&T` is undefined behavior
    *std::ptr::from_ref(num).cast_mut() += 1;
    //~^ ERROR assigning to `&T` is undefined behavior
    *std::ptr::from_ref({ num }).cast_mut() += 1;
    //~^ ERROR assigning to `&T` is undefined behavior
    *{ std::ptr::from_ref(num) }.cast_mut() += 1;
    //~^ ERROR assigning to `&T` is undefined behavior
    *(std::ptr::from_ref({ num }) as *mut i32) += 1;
    //~^ ERROR assigning to `&T` is undefined behavior
    *std::mem::transmute::<_, *mut i32>(num) += 1;
    //~^ ERROR assigning to `&T` is undefined behavior
    *(std::mem::transmute::<_, *mut i32>(num) as *mut i32) += 1;
    //~^ ERROR assigning to `&T` is undefined behavior
    std::ptr::write(
    //~^ ERROR assigning to `&T` is undefined behavior
        std::mem::transmute::<*const i32, *mut i32>(num),
        -1i32,
    );
    *((&std::cell::UnsafeCell::new(0)) as *const _ as *mut i32) = 5;
    //~^ ERROR assigning to `&T` is undefined behavior

    let value = num as *const i32 as *mut i32;
    *value = 1;
    //~^ ERROR assigning to `&T` is undefined behavior
    let value_rebind = value;
    *value_rebind = 1;
    //~^ ERROR assigning to `&T` is undefined behavior
    *(num as *const i32).cast::<i32>().cast_mut() = 2;
    //~^ ERROR assigning to `&T` is undefined behavior
    *(num as *const _ as usize as *mut i32) = 2;
    //~^ ERROR assigning to `&T` is undefined behavior
    std::ptr::write(value, 2);
    //~^ ERROR assigning to `&T` is undefined behavior
    std::ptr::write_unaligned(value, 2);
    //~^ ERROR assigning to `&T` is undefined behavior
    std::ptr::write_volatile(value, 2);
    //~^ ERROR assigning to `&T` is undefined behavior

    unsafe fn generic_assign_to_ref<T>(this: &T, a: T) {
        *(this as *const _ as *mut _) = a;
        //~^ ERROR assigning to `&T` is undefined behavior
    }
}

const RAW_PTR: *mut u8 = 1 as *mut u8;
unsafe fn no_warn() {
    let num = &3i32;
    let mut_num = &mut 3i32;
    let a = &String::from("ffi");

    *(num as *const i32 as *mut i32);
    println!("{}", *(num as *const _ as *const i16));
    println!("{}", *(mut_num as *mut _ as *mut i16));
    ffi(a.as_ptr() as *mut _);
    int_ffi(num as *const _ as *mut _);
    int_ffi(&3 as *const _ as *mut _);
    let mut value = 3;
    let value: *const i32 = &mut value;
    *(value as *const i16 as *mut i16) = 42;
    *RAW_PTR = 42; // RAW_PTR is defined outside the function body,
                   // make sure we don't ICE on it when trying to
                   // determine if we should lint on it or not.
    let cell = &std::cell::UnsafeCell::new(0);
    let _num = &mut *(cell.get() as *mut i32);

    fn safe_as_mut<T>(x: &std::cell::UnsafeCell<T>) -> &mut T {
        unsafe { &mut *std::cell::UnsafeCell::raw_get(x as *const _ as *const _) }
    }

    fn cell_as_mut(x: &std::cell::Cell<i32>) -> &mut i32 {
        unsafe { &mut *std::cell::UnsafeCell::raw_get(x as *const _ as *const _) }
    }

    #[repr(transparent)]
    struct DoesContainUnsafeCell(std::cell::UnsafeCell<i32>);
    fn safe_as_mut2(x: &DoesContainUnsafeCell) -> &mut DoesContainUnsafeCell {
        unsafe { &mut *std::cell::UnsafeCell::raw_get(x as *const _ as *const _) }
    }
}

fn main() {}
