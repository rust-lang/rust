#[doc = "Unsafe pointer utility functions"];

export addr_of;
export mut_addr_of;
export offset;
export const_offset;
export mut_offset;
export null;
export is_null;
export is_not_null;
export memcpy;
export memmove;
export memset;
export buf_len;
export position;
export extensions;

import libc::{c_void, size_t};

#[nolink]
#[abi = "cdecl"]
native mod libc_ {
    #[rust_stack]
    fn memcpy(dest: *c_void, src: *c_void, n: libc::size_t) -> *c_void;
    #[rust_stack]
    fn memmove(dest: *c_void, src: *c_void, n: libc::size_t) -> *c_void;
    #[rust_stack]
    fn memset(dest: *c_void, c: libc::c_int, len: libc::size_t) -> *c_void;
}

#[abi = "rust-intrinsic"]
native mod rusti {
    fn addr_of<T>(val: T) -> *T;
}

#[doc = "Get an unsafe pointer to a value"]
#[inline(always)]
pure fn addr_of<T>(val: T) -> *T { unchecked { rusti::addr_of(val) } }

#[doc = "Get an unsafe mut pointer to a value"]
#[inline(always)]
pure fn mut_addr_of<T>(val: T) -> *mut T {
    unsafe {
        unsafe::reinterpret_cast(rusti::addr_of(val))
    }
}

#[doc = "Calculate the offset from a pointer"]
#[inline(always)]
fn offset<T>(ptr: *T, count: uint) -> *T {
    unsafe {
        (ptr as uint + count * sys::size_of::<T>()) as *T
    }
}

#[doc = "Calculate the offset from a const pointer"]
#[inline(always)]
fn const_offset<T>(ptr: *const T, count: uint) -> *const T {
    unsafe {
        (ptr as uint + count * sys::size_of::<T>()) as *T
    }
}

#[doc = "Calculate the offset from a mut pointer"]
#[inline(always)]
fn mut_offset<T>(ptr: *mut T, count: uint) -> *mut T {
    (ptr as uint + count * sys::size_of::<T>()) as *mut T
}

#[doc = "Return the offset of the first null pointer in `buf`."]
#[inline(always)]
unsafe fn buf_len<T>(buf: **T) -> uint {
    position(buf) {|i| i == null() }
}

#[doc = "Return the first offset `i` such that `f(buf[i]) == true`."]
#[inline(always)]
unsafe fn position<T>(buf: *T, f: fn(T) -> bool) -> uint {
    let mut i = 0u;
    loop {
        if f(*offset(buf, i)) { ret i; }
        else { i += 1u; }
    }
}

#[doc = "Create an unsafe null pointer"]
#[inline(always)]
pure fn null<T>() -> *T { unsafe { unsafe::reinterpret_cast(0u) } }

#[doc = "Returns true if the pointer is equal to the null pointer."]
pure fn is_null<T>(ptr: *const T) -> bool { ptr == null() }

#[doc = "Returns true if the pointer is not equal to the null pointer."]
pure fn is_not_null<T>(ptr: *const T) -> bool { !is_null(ptr) }

#[doc = "
Copies data from one location to another

Copies `count` elements (not bytes) from `src` to `dst`. The source
and destination may not overlap.
"]
#[inline(always)]
unsafe fn memcpy<T>(dst: *T, src: *T, count: uint) {
    let n = count * sys::size_of::<T>();
    libc_::memcpy(dst as *c_void, src as *c_void, n as size_t);
}

#[doc = "
Copies data from one location to another

Copies `count` elements (not bytes) from `src` to `dst`. The source
and destination may overlap.
"]
#[inline(always)]
unsafe fn memmove<T>(dst: *T, src: *T, count: uint)  {
    let n = count * sys::size_of::<T>();
    libc_::memmove(dst as *c_void, src as *c_void, n as size_t);
}

#[inline(always)]
unsafe fn memset<T>(dst: *mut T, c: int, count: uint)  {
    let n = count * sys::size_of::<T>();
    libc_::memset(dst as *c_void, c as libc::c_int, n as size_t);
}

#[doc = "Extension methods for pointers"]
impl extensions<T> for *T {
    #[doc = "Returns true if the pointer is equal to the null pointer."]
    pure fn is_null() -> bool { is_null(self) }

    #[doc = "Returns true if the pointer is not equal to the null pointer."]
    pure fn is_not_null() -> bool { is_not_null(self) }
}

#[test]
fn test() {
    unsafe {
        type pair = {mut fst: int, mut snd: int};
        let p = {mut fst: 10, mut snd: 20};
        let pptr: *mut pair = mut_addr_of(p);
        let iptr: *mut int = unsafe::reinterpret_cast(pptr);
        assert (*iptr == 10);;
        *iptr = 30;
        assert (*iptr == 30);
        assert (p.fst == 30);;

        *pptr = {mut fst: 50, mut snd: 60};
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
}

#[test]
fn test_position() {
    import str::as_c_str;
    import libc::c_char;

    let s = "hello";
    unsafe {
        assert 2u == as_c_str(s) {|p| position(p) {|c| c == 'l' as c_char} };
        assert 4u == as_c_str(s) {|p| position(p) {|c| c == 'o' as c_char} };
        assert 5u == as_c_str(s) {|p| position(p) {|c| c == 0 as c_char } };
    }
}

#[test]
fn test_buf_len() {
    let s0 = "hello";
    let s1 = "there";
    let s2 = "thing";
    str::as_c_str(s0) {|p0|
        str::as_c_str(s1) {|p1|
            str::as_c_str(s2) {|p2|
                let v = [p0, p1, p2, null()];
                vec::as_buf(v) {|vp|
                    assert unsafe { buf_len(vp) } == 3u;
                }
            }
        }
    }
}
