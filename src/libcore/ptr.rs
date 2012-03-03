/*
Module: ptr

Unsafe pointer utility functions
*/
#[abi = "rust-intrinsic"]
native mod rusti {
    fn addr_of<T>(val: T) -> *T;
    fn ptr_offset<T>(ptr: *T, count: ctypes::uintptr_t) -> *T;
    fn memcpy<T>(dst: *T, src: *T, count: ctypes::uintptr_t);
    fn memmove<T>(dst: *T, src: *T, count: ctypes::uintptr_t);
}

/*
Function: addr_of

Get an unsafe pointer to a value
*/
fn addr_of<T>(val: T) -> *T { ret rusti::addr_of(val); }

/*
Function: mut_addr_of

Get an unsafe mutable pointer to a value
*/
fn mut_addr_of<T>(val: T) -> *mutable T unsafe {
    ret unsafe::reinterpret_cast(rusti::addr_of(val));
}

/*
Function: offset

Calculate the offset from a pointer
*/
#[inline(always)]
fn offset<T>(ptr: *T, count: uint) -> *T {
    ret rusti::ptr_offset(ptr, count);
}

/*
Function: mut_offset

Calculate the offset from a mutable pointer
*/
fn mut_offset<T>(ptr: *mutable T, count: uint) -> *mutable T {
    ret rusti::ptr_offset(ptr as *T, count) as *mutable T;
}


/*
Function: null

Create an unsafe null pointer
*/
fn null<T>() -> *T unsafe { ret unsafe::reinterpret_cast(0u); }

/*
Function: memcpy

Copies data from one src to dst that is not overlapping each other.
Count is the number of elements to copy and not the number of bytes.
*/
unsafe fn memcpy<T>(dst: *T, src: *T, count: uint) {
    rusti::memcpy(dst, src, count);
}

/*
Function: memmove

Copies data from one src to dst, overlap between the two pointers may occur.
Count is the number of elements to copy and not the number of bytes.
*/
unsafe fn memmove<T>(dst: *T, src: *T, count: uint)  {
    rusti::memmove(dst, src, count);
}

#[test]
fn test() unsafe {
    type pair = {mutable fst: int, mutable snd: int};
    let p = {mutable fst: 10, mutable snd: 20};
    let pptr: *mutable pair = mut_addr_of(p);
    let iptr: *mutable int = unsafe::reinterpret_cast(pptr);
    assert (*iptr == 10);;
    *iptr = 30;
    assert (*iptr == 30);
    assert (p.fst == 30);;

    *pptr = {mutable fst: 50, mutable snd: 60};
    assert (*iptr == 50);
    assert (p.fst == 50);
    assert (p.snd == 60);

    let v0 = [32000u16, 32001u16, 32002u16];
    let v1 = [0u16, 0u16, 0u16];

    ptr::memcpy(ptr::offset(vec::unsafe::to_ptr(v1), 1u),
                ptr::offset(vec::unsafe::to_ptr(v0), 1u), 1u);
    assert (v1[0] == 0u16 && v1[1] == 32001u16 && v1[2] == 0u16);
    ptr::memcpy(vec::unsafe::to_ptr(v1),
                ptr::offset(vec::unsafe::to_ptr(v0), 2u), 1u);
    assert (v1[0] == 32002u16 && v1[1] == 32001u16 && v1[2] == 0u16);
    ptr::memcpy(ptr::offset(vec::unsafe::to_ptr(v1), 2u),
                vec::unsafe::to_ptr(v0), 1u);
    assert (v1[0] == 32002u16 && v1[1] == 32001u16 && v1[2] == 32000u16);
}