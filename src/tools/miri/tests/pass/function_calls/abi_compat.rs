use std::num;
use std::mem;

fn test_abi_compat<T, U>(t: T, u: U) {
    fn id<T>(x: T) -> T { x }
    
    // This checks ABI compatibility both for arguments and return values,
    // in both directions.
    let f: fn(T) -> T = id;
    let f: fn(U) -> U = unsafe { std::mem::transmute(f) };
    drop(f(u));
    
    let f: fn(U) -> U = id;
    let f: fn(T) -> T = unsafe { std::mem::transmute(f) };
    drop(f(t));
}

fn main() {
    test_abi_compat(0u32, 'x');
    test_abi_compat(&0u32, &([true; 4], [0u32; 0]));
    test_abi_compat(0u32, mem::MaybeUninit::new(0u32));
    test_abi_compat(42u32, num::NonZeroU32::new(1).unwrap());
    test_abi_compat(0u32, Some(num::NonZeroU32::new(1).unwrap()));
    test_abi_compat(0u32, 0i32);
    // Note that `bool` and `u8` are *not* compatible!
    // One of them has `arg_ext: Zext`, the other does not.
}
