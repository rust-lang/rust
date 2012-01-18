/*
Module: ptr

Unsafe pointer utility functions
*/
#[abi = "rust-intrinsic"]
native mod rusti {
    fn addr_of<T>(val: T) -> *T;
    fn ptr_offset<T>(ptr: *T, count: uint) -> *T;
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
}