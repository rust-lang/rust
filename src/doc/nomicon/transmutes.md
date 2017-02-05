% Transmutes

Get out of our way type system! We're going to reinterpret these bits or die
trying! Even though this book is all about doing things that are unsafe, I
really can't emphasize that you should deeply think about finding Another Way
than the operations covered in this section. This is really, truly, the most
horribly unsafe thing you can do in Rust. The railguards here are dental floss.

`mem::transmute<T, U>` takes a value of type `T` and reinterprets it to have
type `U`. The only restriction is that the `T` and `U` are verified to have the
same size. The ways to cause Undefined Behavior with this are mind boggling.

* First and foremost, creating an instance of *any* type with an invalid state
  is going to cause arbitrary chaos that can't really be predicted.
* Transmute has an overloaded return type. If you do not specify the return type
  it may produce a surprising type to satisfy inference.
* Making a primitive with an invalid value is UB
* Transmuting between non-repr(C) types is UB
    * However, a transmute to a type and back may be defined for all T, with a
      few restraints on U.
        * U must have no invalid bit patterns (i.e., `(bool)0x3` is an invalid
          bit pattern)
        * U must have no padding
        * U must be `#[repr(C)]`
    * This is useful for, for example, transmuting to a
      `[u8; std::mem::size_of<T>]`, and back
* Transmuting an & to &mut is UB
    * Transmuting an & to &mut is *always* UB
    * No you can't do it
    * No you're not special
* Transmuting to a reference without an explicitly provided lifetime
  produces an [unbounded lifetime]

`mem::transmute_copy<T, U>` somehow manages to be *even more* wildly unsafe than
this. It copies `size_of<U>` bytes out of an `&T` and interprets them as a `U`.
The size check that `mem::transmute` has is gone (as it may be valid to copy
out a prefix), though it is Undefined Behavior for `U` to be larger than `T`.

Also of course you can get most of the functionality of these functions using
pointer casts and `std::ptr::copy_nonoverlapping`.


[unbounded lifetime]: unbounded-lifetimes.html
