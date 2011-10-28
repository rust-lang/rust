/*
Module: unsafe

Unsafe operations
*/

native "rust-intrinsic" mod rusti {
    fn cast<T, U>(src: T) -> U;
}

native "c-stack-cdecl" mod rustrt {
    fn leak<T>(-thing: T);
}

/*
Function: reinterpret_cast

Casts the value at `src` to U. The two types must have the same length.
*/
unsafe fn reinterpret_cast<T, U>(src: T) -> U { ret rusti::cast(src); }

/*
Function: leak

Move `thing` into the void.

The leak function will take ownership of the provided value but neglect
to run any required cleanup or memory-management operations on it. This
can be used for various acts of magick, particularly when using
reinterpret_cast on managed pointer types.
*/
unsafe fn leak<T>(-thing: T) { rustrt::leak(thing); }
