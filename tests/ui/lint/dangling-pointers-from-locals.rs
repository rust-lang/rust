//@ check-pass

struct Zst((), ());
struct Adt(u8);

const X: u8 = 5;

fn simple() -> *const u8 {
    let x = 0;
    &x
    //~^ WARN dangling pointer
}

fn bindings() -> *const u8 {
    let x = 0;
    let x = &x;
    x
    //~^ WARN dangling pointer
}

fn bindings_with_return() -> *const u8 {
    let x = 42;
    let y = &x;
    return y;
    //~^ WARN dangling pointer
}

fn with_simple_cast() -> *const u8 {
    let x = 0u8;
    &x as *const u8
    //~^ WARN dangling pointer
}

fn bindings_and_casts() -> *const u8 {
    let x = 0u8;
    let x = &x as *const u8;
    x as *const u8
    //~^ WARN dangling pointer
}

fn return_with_complex_cast() -> *mut u8 {
    let mut x = 0u8;
    return &mut x as *mut u8 as *const u8 as *mut u8;
    //~^ WARN dangling pointer
}

fn with_block() -> *const u8 {
    let x = 0;
    &{ x }
    //~^ WARN dangling pointer
}

fn with_many_blocks() -> *const u8 {
    let x = 0;
    {
        {
            &{
                //~^ WARN dangling pointer
                { x }
            }
        }
    }
}

fn simple_return() -> *const u8 {
    let x = 0;
    return &x;
    //~^ WARN dangling pointer
}

fn return_mut() -> *mut u8 {
    let mut x = 0;
    return &mut x;
    //~^ WARN dangling pointer
}

fn const_and_flow() -> *const u8 {
    if false {
        let x = 8;
        return &x;
        //~^ WARN dangling pointer
    }
    &X // not dangling
}

fn vector<T: Default>() -> *const Vec<T> {
    let x = vec![T::default()];
    &x
    //~^ WARN dangling pointer
}

fn local_adt() -> *const Adt {
    let x = Adt(5);
    return &x;
    //~^ WARN dangling pointer
}

fn closure() -> *const u8 {
    let _x = || -> *const u8 {
        let x = 8;
        return &x;
        //~^ WARN dangling pointer
    };
    &X // not dangling
}

fn fn_ptr() -> *const fn() -> u8 {
    fn ret_u8() -> u8 {
        0
    }

    let x = ret_u8 as fn() -> u8;
    &x
    //~^ WARN dangling pointer
}

fn as_arg(a: Adt) -> *const Adt {
    &a
    //~^ WARN dangling pointer
}

fn fn_ptr_as_arg(a: fn() -> u8) -> *const fn() -> u8 {
    &a
    //~^ WARN dangling pointer
}

fn ptr_as_arg(a: *const Adt) -> *const *const Adt {
    &a
    //~^ WARN dangling pointer
}

fn adt_as_arg(a: &Adt) -> *const &Adt {
    &a
    //~^ WARN dangling pointer
}

fn unit() -> *const () {
    let x = ();
    &x // not dangling
}

fn zst() -> *const Zst {
    let x = Zst((), ());
    &x // not dangling
}

fn ref_implicit(a: &Adt) -> *const Adt {
    a // not dangling
}

fn ref_explicit(a: &Adt) -> *const Adt {
    &*a // not dangling
}

fn identity(a: *const Adt) -> *const Adt {
    a // not dangling
}

fn from_ref(a: &Adt) -> *const Adt {
    std::ptr::from_ref(a) // not dangling
}

fn inner_static() -> *const u8 {
    static U: u8 = 5;
    if false {
        return &U as *const u8; // not dangling
    }
    &U // not dangling
}

fn return_in_closure() {
    let x = 0;
    let c = || -> *const u8 {
        &x // not dangling by it-self
    };
}

fn option<T: Default>() -> *const Option<T> {
    let x = Some(T::default());
    &x // can't compute layout of `Option<T>`, so cnat' be sure it won't be a ZST
}

fn generic<T: Default>() -> *const T {
    let x = T::default();
    &x // can't compute layout of `T`, so can't be sure it won't be a ZST
}

fn main() {}
