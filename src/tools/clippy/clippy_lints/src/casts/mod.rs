mod as_pointer_underscore;
mod as_ptr_cast_mut;
mod as_underscore;
mod borrow_as_ptr;
mod cast_abs_to_unsigned;
mod cast_enum_constructor;
mod cast_lossless;
mod cast_nan_to_int;
mod cast_possible_truncation;
mod cast_possible_wrap;
mod cast_precision_loss;
mod cast_ptr_alignment;
mod cast_sign_loss;
mod cast_slice_different_sizes;
mod cast_slice_from_raw_parts;
mod char_lit_as_u8;
mod confusing_method_to_numeric_cast;
mod fn_to_numeric_cast;
mod fn_to_numeric_cast_any;
mod fn_to_numeric_cast_with_truncation;
mod manual_dangling_ptr;
mod ptr_as_ptr;
mod ptr_cast_constness;
mod ref_as_ptr;
mod unnecessary_cast;
mod utils;
mod zero_ptr;

use clippy_config::Conf;
use clippy_utils::is_hir_ty_cfg_dependant;
use clippy_utils::msrvs::{self, Msrv};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::impl_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts from any numeric type to a float type where
    /// the receiving type cannot store all values from the original type without
    /// rounding errors. This possible rounding is to be expected, so this lint is
    /// `Allow` by default.
    ///
    /// Basically, this warns on casting any integer with 32 or more bits to `f32`
    /// or any 64-bit integer to `f64`.
    ///
    /// ### Why is this bad?
    /// It's not bad at all. But in some applications it can be
    /// helpful to know where precision loss can take place. This lint can help find
    /// those places in the code.
    ///
    /// ### Example
    /// ```no_run
    /// let x = u64::MAX;
    /// x as f64;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub CAST_PRECISION_LOSS,
    pedantic,
    "casts that cause loss of precision, e.g., `x as f32` where `x: u64`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts from a signed to an unsigned numeric
    /// type. In this case, negative values wrap around to large positive values,
    /// which can be quite surprising in practice. However, since the cast works as
    /// defined, this lint is `Allow` by default.
    ///
    /// ### Why is this bad?
    /// Possibly surprising results. You can activate this lint
    /// as a one-time check to see where numeric wrapping can arise.
    ///
    /// ### Example
    /// ```no_run
    /// let y: i8 = -1;
    /// y as u64; // will return 18446744073709551615
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub CAST_SIGN_LOSS,
    pedantic,
    "casts from signed types to unsigned types, e.g., `x as u32` where `x: i32`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts between numeric types that may
    /// truncate large values. This is expected behavior, so the cast is `Allow` by
    /// default. It suggests user either explicitly ignore the lint,
    /// or use `try_from()` and handle the truncation, default, or panic explicitly.
    ///
    /// ### Why is this bad?
    /// In some problem domains, it is good practice to avoid
    /// truncation. This lint can be activated to help assess where additional
    /// checks could be beneficial.
    ///
    /// ### Example
    /// ```no_run
    /// fn as_u8(x: u64) -> u8 {
    ///     x as u8
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn as_u8(x: u64) -> u8 {
    ///     if let Ok(x) = u8::try_from(x) {
    ///         x
    ///     } else {
    ///         todo!();
    ///     }
    /// }
    /// // Or
    /// #[allow(clippy::cast_possible_truncation)]
    /// fn as_u16(x: u64) -> u16 {
    ///     x as u16
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub CAST_POSSIBLE_TRUNCATION,
    pedantic,
    "casts that may cause truncation of the value, e.g., `x as u8` where `x: u32`, or `x as i32` where `x: f32`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts from an unsigned type to a signed type of
    /// the same size, or possibly smaller due to target-dependent integers.
    /// Performing such a cast is a no-op for the compiler (that is, nothing is
    /// changed at the bit level), and the binary representation of the value is
    /// reinterpreted. This can cause wrapping if the value is too big
    /// for the target signed type. However, the cast works as defined, so this lint
    /// is `Allow` by default.
    ///
    /// ### Why is this bad?
    /// While such a cast is not bad in itself, the results can
    /// be surprising when this is not the intended behavior:
    ///
    /// ### Example
    /// ```no_run
    /// let _ = u32::MAX as i32; // will yield a value of `-1`
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let _ = i32::try_from(u32::MAX).ok();
    /// ```
    ///
    #[clippy::version = "pre 1.29.0"]
    pub CAST_POSSIBLE_WRAP,
    pedantic,
    "casts that may cause wrapping around the value, e.g., `x as i32` where `x: u32` and `x > i32::MAX`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts between numeric types that can be replaced by safe
    /// conversion functions.
    ///
    /// ### Why is this bad?
    /// Rust's `as` keyword will perform many kinds of conversions, including
    /// silently lossy conversions. Conversion functions such as `i32::from`
    /// will only perform lossless conversions. Using the conversion functions
    /// prevents conversions from becoming silently lossy if the input types
    /// ever change, and makes it clear for people reading the code that the
    /// conversion is lossless.
    ///
    /// ### Example
    /// ```no_run
    /// fn as_u64(x: u8) -> u64 {
    ///     x as u64
    /// }
    /// ```
    ///
    /// Using `::from` would look like this:
    ///
    /// ```no_run
    /// fn as_u64(x: u8) -> u64 {
    ///     u64::from(x)
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub CAST_LOSSLESS,
    pedantic,
    "casts using `as` that are known to be lossless, e.g., `x as u64` where `x: u8`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts to the same type, casts of int literals to integer
    /// types, casts of float literals to float types, and casts between raw
    /// pointers that don't change type or constness.
    ///
    /// ### Why is this bad?
    /// It's just unnecessary.
    ///
    /// ### Known problems
    /// When the expression on the left is a function call, the lint considers
    /// the return type to be a type alias if it's aliased through a `use`
    /// statement (like `use std::io::Result as IoResult`). It will not lint
    /// such cases.
    ///
    /// This check will only work on primitive types without any intermediate
    /// references: raw pointers and trait objects may or may not work.
    ///
    /// ### Example
    /// ```no_run
    /// let _ = 2i32 as i32;
    /// let _ = 0.5 as f32;
    /// ```
    ///
    /// Better:
    ///
    /// ```no_run
    /// let _ = 2_i32;
    /// let _ = 0.5_f32;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub UNNECESSARY_CAST,
    complexity,
    "cast to the same type, e.g., `x as i32` where `x: i32`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts, using `as` or `pointer::cast`, from a
    /// less strictly aligned pointer to a more strictly aligned pointer.
    ///
    /// ### Why is this bad?
    /// Dereferencing the resulting pointer may be undefined behavior.
    ///
    /// ### Known problems
    /// Using [`std::ptr::read_unaligned`](https://doc.rust-lang.org/std/ptr/fn.read_unaligned.html) and [`std::ptr::write_unaligned`](https://doc.rust-lang.org/std/ptr/fn.write_unaligned.html) or
    /// similar on the resulting pointer is fine. Is over-zealous: casts with
    /// manual alignment checks or casts like `u64` -> `u8` -> `u16` can be
    /// fine. Miri is able to do a more in-depth analysis.
    ///
    /// ### Example
    /// ```no_run
    /// let _ = (&1u8 as *const u8) as *const u16;
    /// let _ = (&mut 1u8 as *mut u8) as *mut u16;
    ///
    /// (&1u8 as *const u8).cast::<u16>();
    /// (&mut 1u8 as *mut u8).cast::<u16>();
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub CAST_PTR_ALIGNMENT,
    pedantic,
    "cast from a pointer to a more strictly aligned pointer"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts of function pointers to something other than `usize`.
    ///
    /// ### Why is this bad?
    /// Casting a function pointer to anything other than `usize`/`isize` is
    /// not portable across architectures. If the target type is too small the
    /// address would be truncated, and target types larger than `usize` are
    /// unnecessary.
    ///
    /// Casting to `isize` also doesn't make sense, since addresses are never
    /// signed.
    ///
    /// ### Example
    /// ```no_run
    /// fn fun() -> i32 { 1 }
    /// let _ = fun as i64;
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # fn fun() -> i32 { 1 }
    /// let _ = fun as usize;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub FN_TO_NUMERIC_CAST,
    style,
    "casting a function pointer to a numeric type other than `usize`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts of a function pointer to a numeric type not wide enough to
    /// store an address.
    ///
    /// ### Why is this bad?
    /// Such a cast discards some bits of the function's address. If this is intended, it would be more
    /// clearly expressed by casting to `usize` first, then casting the `usize` to the intended type (with
    /// a comment) to perform the truncation.
    ///
    /// ### Example
    /// ```no_run
    /// fn fn1() -> i16 {
    ///     1
    /// };
    /// let _ = fn1 as i32;
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// // Cast to usize first, then comment with the reason for the truncation
    /// fn fn1() -> i16 {
    ///     1
    /// };
    /// let fn_ptr = fn1 as usize;
    /// let fn_ptr_truncated = fn_ptr as i32;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub FN_TO_NUMERIC_CAST_WITH_TRUNCATION,
    style,
    "casting a function pointer to a numeric type not wide enough to store the address"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts of a function pointer to any integer type.
    ///
    /// ### Why restrict this?
    /// Casting a function pointer to an integer can have surprising results and can occur
    /// accidentally if parentheses are omitted from a function call. If you aren't doing anything
    /// low-level with function pointers then you can opt out of casting functions to integers in
    /// order to avoid mistakes. Alternatively, you can use this lint to audit all uses of function
    /// pointer casts in your code.
    ///
    /// ### Example
    /// ```no_run
    /// // fn1 is cast as `usize`
    /// fn fn1() -> u16 {
    ///     1
    /// };
    /// let _ = fn1 as usize;
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// // maybe you intended to call the function?
    /// fn fn2() -> u16 {
    ///     1
    /// };
    /// let _ = fn2() as usize;
    ///
    /// // or
    ///
    /// // maybe you intended to cast it to a function type?
    /// fn fn3() -> u16 {
    ///     1
    /// }
    /// let _ = fn3 as fn() -> u16;
    /// ```
    #[clippy::version = "1.58.0"]
    pub FN_TO_NUMERIC_CAST_ANY,
    restriction,
    "casting a function pointer to any integer type"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for expressions where a character literal is cast
    /// to `u8` and suggests using a byte literal instead.
    ///
    /// ### Why is this bad?
    /// In general, casting values to smaller types is
    /// error-prone and should be avoided where possible. In the particular case of
    /// converting a character literal to `u8`, it is easy to avoid by just using a
    /// byte literal instead. As an added bonus, `b'a'` is also slightly shorter
    /// than `'a' as u8`.
    ///
    /// ### Example
    /// ```rust,ignore
    /// 'x' as u8
    /// ```
    ///
    /// A better version, using the byte literal:
    ///
    /// ```rust,ignore
    /// b'x'
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub CHAR_LIT_AS_U8,
    complexity,
    "casting a character literal to `u8` truncates"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `as` casts between raw pointers that don't change their
    /// constness, namely `*const T` to `*const U` and `*mut T` to `*mut U`.
    ///
    /// ### Why is this bad?
    /// Though `as` casts between raw pointers are not terrible,
    /// `pointer::cast` is safer because it cannot accidentally change the
    /// pointer's mutability, nor cast the pointer to other types like `usize`.
    ///
    /// ### Example
    /// ```no_run
    /// let ptr: *const u32 = &42_u32;
    /// let mut_ptr: *mut u32 = &mut 42_u32;
    /// let _ = ptr as *const i32;
    /// let _ = mut_ptr as *mut i32;
    /// ```
    /// Use instead:
    /// ```no_run
    /// let ptr: *const u32 = &42_u32;
    /// let mut_ptr: *mut u32 = &mut 42_u32;
    /// let _ = ptr.cast::<i32>();
    /// let _ = mut_ptr.cast::<i32>();
    /// ```
    #[clippy::version = "1.51.0"]
    pub PTR_AS_PTR,
    pedantic,
    "casting using `as` between raw pointers that doesn't change their constness, where `pointer::cast` could take the place of `as`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `as` casts between raw pointers that change their constness, namely `*const T` to
    /// `*mut T` and `*mut T` to `*const T`.
    ///
    /// ### Why is this bad?
    /// Though `as` casts between raw pointers are not terrible, `pointer::cast_mut` and
    /// `pointer::cast_const` are safer because they cannot accidentally cast the pointer to another
    /// type. Or, when null pointers are involved, `null()` and `null_mut()` can be used directly.
    ///
    /// ### Example
    /// ```no_run
    /// let ptr: *const u32 = &42_u32;
    /// let mut_ptr = ptr as *mut u32;
    /// let ptr = mut_ptr as *const u32;
    /// let ptr1 = std::ptr::null::<u32>() as *mut u32;
    /// let ptr2 = std::ptr::null_mut::<u32>() as *const u32;
    /// let ptr3 = std::ptr::null::<u32>().cast_mut();
    /// let ptr4 = std::ptr::null_mut::<u32>().cast_const();
    /// ```
    /// Use instead:
    /// ```no_run
    /// let ptr: *const u32 = &42_u32;
    /// let mut_ptr = ptr.cast_mut();
    /// let ptr = mut_ptr.cast_const();
    /// let ptr1 = std::ptr::null_mut::<u32>();
    /// let ptr2 = std::ptr::null::<u32>();
    /// let ptr3 = std::ptr::null_mut::<u32>();
    /// let ptr4 = std::ptr::null::<u32>();
    /// ```
    #[clippy::version = "1.72.0"]
    pub PTR_CAST_CONSTNESS,
    pedantic,
    "casting using `as` on raw pointers to change constness when specialized methods apply"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts from an enum type to an integral type that will definitely truncate the
    /// value.
    ///
    /// ### Why is this bad?
    /// The resulting integral value will not match the value of the variant it came from.
    ///
    /// ### Example
    /// ```no_run
    /// enum E { X = 256 };
    /// let _ = E::X as u8;
    /// ```
    #[clippy::version = "1.61.0"]
    pub CAST_ENUM_TRUNCATION,
    suspicious,
    "casts from an enum type to an integral type that will truncate the value"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `as` casts between raw pointers to slices with differently sized elements.
    ///
    /// ### Why is this bad?
    /// The produced raw pointer to a slice does not update its length metadata. The produced
    /// pointer will point to a different number of bytes than the original pointer because the
    /// length metadata of a raw slice pointer is in elements rather than bytes.
    /// Producing a slice reference from the raw pointer will either create a slice with
    /// less data (which can be surprising) or create a slice with more data and cause Undefined Behavior.
    ///
    /// ### Example
    /// // Missing data
    /// ```no_run
    /// let a = [1_i32, 2, 3, 4];
    /// let p = &a as *const [i32] as *const [u8];
    /// unsafe {
    ///     println!("{:?}", &*p);
    /// }
    /// ```
    /// // Undefined Behavior (note: also potential alignment issues)
    /// ```no_run
    /// let a = [1_u8, 2, 3, 4];
    /// let p = &a as *const [u8] as *const [u32];
    /// unsafe {
    ///     println!("{:?}", &*p);
    /// }
    /// ```
    /// Instead use `ptr::slice_from_raw_parts` to construct a slice from a data pointer and the correct length
    /// ```no_run
    /// let a = [1_i32, 2, 3, 4];
    /// let old_ptr = &a as *const [i32];
    /// // The data pointer is cast to a pointer to the target `u8` not `[u8]`
    /// // The length comes from the known length of 4 i32s times the 4 bytes per i32
    /// let new_ptr = core::ptr::slice_from_raw_parts(old_ptr as *const u8, 16);
    /// unsafe {
    ///     println!("{:?}", &*new_ptr);
    /// }
    /// ```
    #[clippy::version = "1.61.0"]
    pub CAST_SLICE_DIFFERENT_SIZES,
    correctness,
    "casting using `as` between raw pointers to slices of types with different sizes"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts from an enum tuple constructor to an integer.
    ///
    /// ### Why is this bad?
    /// The cast is easily confused with casting a c-like enum value to an integer.
    ///
    /// ### Example
    /// ```no_run
    /// enum E { X(i32) };
    /// let _ = E::X as usize;
    /// ```
    #[clippy::version = "1.61.0"]
    pub CAST_ENUM_CONSTRUCTOR,
    suspicious,
    "casts from an enum tuple constructor to an integer"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of the `abs()` method that cast the result to unsigned.
    ///
    /// ### Why is this bad?
    /// The `unsigned_abs()` method avoids panic when called on the MIN value.
    ///
    /// ### Example
    /// ```no_run
    /// let x: i32 = -42;
    /// let y: u32 = x.abs() as u32;
    /// ```
    /// Use instead:
    /// ```no_run
    /// let x: i32 = -42;
    /// let y: u32 = x.unsigned_abs();
    /// ```
    #[clippy::version = "1.62.0"]
    pub CAST_ABS_TO_UNSIGNED,
    suspicious,
    "casting the result of `abs()` to an unsigned integer can panic"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the usage of `as _` conversion using inferred type.
    ///
    /// ### Why restrict this?
    /// The conversion might include lossy conversion or a dangerous cast that might go
    /// undetected due to the type being inferred.
    ///
    /// The lint is allowed by default as using `_` is less wordy than always specifying the type.
    ///
    /// ### Example
    /// ```no_run
    /// fn foo(n: usize) {}
    /// let n: u16 = 256;
    /// foo(n as _);
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn foo(n: usize) {}
    /// let n: u16 = 256;
    /// foo(n as usize);
    /// ```
    #[clippy::version = "1.63.0"]
    pub AS_UNDERSCORE,
    restriction,
    "detects `as _` conversion"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the usage of `&expr as *const T` or
    /// `&mut expr as *mut T`, and suggest using `&raw const` or
    /// `&raw mut` instead.
    ///
    /// ### Why is this bad?
    /// This would improve readability and avoid creating a reference
    /// that points to an uninitialized value or unaligned place.
    /// Read the `&raw` explanation in the Reference for more information.
    ///
    /// ### Example
    /// ```no_run
    /// let val = 1;
    /// let p = &val as *const i32;
    ///
    /// let mut val_mut = 1;
    /// let p_mut = &mut val_mut as *mut i32;
    /// ```
    /// Use instead:
    /// ```no_run
    /// let val = 1;
    /// let p = &raw const val;
    ///
    /// let mut val_mut = 1;
    /// let p_mut = &raw mut val_mut;
    /// ```
    #[clippy::version = "1.60.0"]
    pub BORROW_AS_PTR,
    pedantic,
    "borrowing just to cast to a raw pointer"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for a raw slice being cast to a slice pointer
    ///
    /// ### Why is this bad?
    /// This can result in multiple `&mut` references to the same location when only a pointer is
    /// required.
    /// `ptr::slice_from_raw_parts` is a safe alternative that doesn't require
    /// the same [safety requirements] to be upheld.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let _: *const [u8] = std::slice::from_raw_parts(ptr, len) as *const _;
    /// let _: *mut [u8] = std::slice::from_raw_parts_mut(ptr, len) as *mut _;
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// let _: *const [u8] = std::ptr::slice_from_raw_parts(ptr, len);
    /// let _: *mut [u8] = std::ptr::slice_from_raw_parts_mut(ptr, len);
    /// ```
    /// [safety requirements]: https://doc.rust-lang.org/std/slice/fn.from_raw_parts.html#safety
    #[clippy::version = "1.65.0"]
    pub CAST_SLICE_FROM_RAW_PARTS,
    suspicious,
    "casting a slice created from a pointer and length to a slice pointer"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the result of a `&self`-taking `as_ptr` being cast to a mutable pointer.
    ///
    /// ### Why is this bad?
    /// Since `as_ptr` takes a `&self`, the pointer won't have write permissions unless interior
    /// mutability is used, making it unlikely that having it as a mutable pointer is correct.
    ///
    /// ### Example
    /// ```no_run
    /// let mut vec = Vec::<u8>::with_capacity(1);
    /// let ptr = vec.as_ptr() as *mut u8;
    /// unsafe { ptr.write(4) }; // UNDEFINED BEHAVIOUR
    /// ```
    /// Use instead:
    /// ```no_run
    /// let mut vec = Vec::<u8>::with_capacity(1);
    /// let ptr = vec.as_mut_ptr();
    /// unsafe { ptr.write(4) };
    /// ```
    #[clippy::version = "1.66.0"]
    pub AS_PTR_CAST_MUT,
    nursery,
    "casting the result of the `&self`-taking `as_ptr` to a mutable pointer"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for a known NaN float being cast to an integer
    ///
    /// ### Why is this bad?
    /// NaNs are cast into zero, so one could simply use this and make the
    /// code more readable. The lint could also hint at a programmer error.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let _ = (0.0_f32 / 0.0) as u64;
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// let _ = 0_u64;
    /// ```
    #[clippy::version = "1.66.0"]
    pub CAST_NAN_TO_INT,
    suspicious,
    "casting a known floating-point NaN into an integer"
}

declare_clippy_lint! {
    /// ### What it does
    /// Catch casts from `0` to some pointer type
    ///
    /// ### Why is this bad?
    /// This generally means `null` and is better expressed as
    /// {`std`, `core`}`::ptr::`{`null`, `null_mut`}.
    ///
    /// ### Example
    /// ```no_run
    /// let a = 0 as *const u32;
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let a = std::ptr::null::<u32>();
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub ZERO_PTR,
    style,
    "using `0 as *{const, mut} T`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts of references to pointer using `as`
    /// and suggests `std::ptr::from_ref` and `std::ptr::from_mut` instead.
    ///
    /// ### Why is this bad?
    /// Using `as` casts may result in silently changing mutability or type.
    ///
    /// ### Example
    /// ```no_run
    /// let a_ref = &1;
    /// let a_ptr = a_ref as *const _;
    /// ```
    /// Use instead:
    /// ```no_run
    /// let a_ref = &1;
    /// let a_ptr = std::ptr::from_ref(a_ref);
    /// ```
    #[clippy::version = "1.78.0"]
    pub REF_AS_PTR,
    pedantic,
    "using `as` to cast a reference to pointer"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the usage of `as *const _` or `as *mut _` conversion using inferred type.
    ///
    /// ### Why restrict this?
    /// The conversion might include a dangerous cast that might go undetected due to the type being inferred.
    ///
    /// ### Example
    /// ```no_run
    /// fn as_usize<T>(t: &T) -> usize {
    ///     // BUG: `t` is already a reference, so we will here
    ///     // return a dangling pointer to a temporary value instead
    ///     &t as *const _ as usize
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn as_usize<T>(t: &T) -> usize {
    ///     t as *const T as usize
    /// }
    /// ```
    #[clippy::version = "1.85.0"]
    pub AS_POINTER_UNDERSCORE,
    restriction,
    "detects `as *mut _` and `as *const _` conversion"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts of small constant literals or `mem::align_of` results to raw pointers.
    ///
    /// ### Why is this bad?
    /// This creates a dangling pointer and is better expressed as
    /// {`std`, `core`}`::ptr::`{`dangling`, `dangling_mut`}.
    ///
    /// ### Example
    /// ```no_run
    /// let ptr = 4 as *const u32;
    /// let aligned = std::mem::align_of::<u32>() as *const u32;
    /// let mut_ptr: *mut i64 = 8 as *mut _;
    /// ```
    /// Use instead:
    /// ```no_run
    /// let ptr = std::ptr::dangling::<u32>();
    /// let aligned = std::ptr::dangling::<u32>();
    /// let mut_ptr: *mut i64 = std::ptr::dangling_mut();
    /// ```
    #[clippy::version = "1.88.0"]
    pub MANUAL_DANGLING_PTR,
    style,
    "casting small constant literals to pointers to create dangling pointers"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for casts of a primitive method pointer like `max`/`min` to any integer type.
    ///
    /// ### Why restrict this?
    /// Casting a function pointer to an integer can have surprising results and can occur
    /// accidentally if parentheses are omitted from a function call. If you aren't doing anything
    /// low-level with function pointers then you can opt out of casting functions to integers in
    /// order to avoid mistakes. Alternatively, you can use this lint to audit all uses of function
    /// pointer casts in your code.
    ///
    /// ### Example
    /// ```no_run
    /// let _ = u16::max as usize;
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let _ = u16::MAX as usize;
    /// ```
    #[clippy::version = "1.86.0"]
    pub CONFUSING_METHOD_TO_NUMERIC_CAST,
    suspicious,
    "casting a primitive method pointer to any integer type"
}

pub struct Casts {
    msrv: Msrv,
}

impl Casts {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl_lint_pass!(Casts => [
    CAST_PRECISION_LOSS,
    CAST_SIGN_LOSS,
    CAST_POSSIBLE_TRUNCATION,
    CAST_POSSIBLE_WRAP,
    CAST_LOSSLESS,
    CAST_PTR_ALIGNMENT,
    CAST_SLICE_DIFFERENT_SIZES,
    UNNECESSARY_CAST,
    FN_TO_NUMERIC_CAST_ANY,
    FN_TO_NUMERIC_CAST,
    FN_TO_NUMERIC_CAST_WITH_TRUNCATION,
    CHAR_LIT_AS_U8,
    PTR_AS_PTR,
    PTR_CAST_CONSTNESS,
    CAST_ENUM_TRUNCATION,
    CAST_ENUM_CONSTRUCTOR,
    CAST_ABS_TO_UNSIGNED,
    AS_UNDERSCORE,
    BORROW_AS_PTR,
    CAST_SLICE_FROM_RAW_PARTS,
    AS_PTR_CAST_MUT,
    CAST_NAN_TO_INT,
    ZERO_PTR,
    REF_AS_PTR,
    AS_POINTER_UNDERSCORE,
    MANUAL_DANGLING_PTR,
    CONFUSING_METHOD_TO_NUMERIC_CAST,
]);

impl<'tcx> LateLintPass<'tcx> for Casts {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.in_external_macro(cx.sess().source_map()) {
            return;
        }

        if let ExprKind::Cast(cast_from_expr, cast_to_hir) = expr.kind {
            if is_hir_ty_cfg_dependant(cx, cast_to_hir) {
                return;
            }
            let (cast_from, cast_to) = (
                cx.typeck_results().expr_ty(cast_from_expr),
                cx.typeck_results().expr_ty(expr),
            );

            if !expr.span.from_expansion() && unnecessary_cast::check(cx, expr, cast_from_expr, cast_from, cast_to) {
                return;
            }
            cast_slice_from_raw_parts::check(cx, expr, cast_from_expr, cast_to, self.msrv);
            ptr_cast_constness::check(cx, expr, cast_from_expr, cast_from, cast_to, self.msrv);
            as_ptr_cast_mut::check(cx, expr, cast_from_expr, cast_to);
            fn_to_numeric_cast_any::check(cx, expr, cast_from_expr, cast_from, cast_to);
            confusing_method_to_numeric_cast::check(cx, expr, cast_from_expr, cast_from, cast_to);
            fn_to_numeric_cast::check(cx, expr, cast_from_expr, cast_from, cast_to);
            fn_to_numeric_cast_with_truncation::check(cx, expr, cast_from_expr, cast_from, cast_to);
            zero_ptr::check(cx, expr, cast_from_expr, cast_to_hir);

            if self.msrv.meets(cx, msrvs::MANUAL_DANGLING_PTR) {
                manual_dangling_ptr::check(cx, expr, cast_from_expr, cast_to_hir);
            }

            if cast_to.is_numeric() {
                cast_possible_truncation::check(cx, expr, cast_from_expr, cast_from, cast_to, cast_to_hir.span);
                if cast_from.is_numeric() {
                    cast_possible_wrap::check(cx, expr, cast_from, cast_to);
                    cast_precision_loss::check(cx, expr, cast_from, cast_to);
                    cast_sign_loss::check(cx, expr, cast_from_expr, cast_from, cast_to);
                    cast_abs_to_unsigned::check(cx, expr, cast_from_expr, cast_from, cast_to, self.msrv);
                    cast_nan_to_int::check(cx, expr, cast_from_expr, cast_from, cast_to);
                }
                cast_lossless::check(cx, expr, cast_from_expr, cast_from, cast_to, cast_to_hir, self.msrv);
                cast_enum_constructor::check(cx, expr, cast_from_expr, cast_from);
            }

            as_underscore::check(cx, expr, cast_to_hir);
            as_pointer_underscore::check(cx, cast_to, cast_to_hir);

            let was_borrow_as_ptr_emitted = self.msrv.meets(cx, msrvs::BORROW_AS_PTR)
                && borrow_as_ptr::check(cx, expr, cast_from_expr, cast_to_hir, self.msrv);
            if !was_borrow_as_ptr_emitted && self.msrv.meets(cx, msrvs::PTR_FROM_REF) {
                ref_as_ptr::check(cx, expr, cast_from_expr, cast_to_hir);
            }
        }

        if self.msrv.meets(cx, msrvs::RAW_REF_OP) {
            borrow_as_ptr::check_implicit_cast(cx, expr);
        }
        cast_ptr_alignment::check(cx, expr);
        char_lit_as_u8::check(cx, expr);
        ptr_as_ptr::check(cx, expr, self.msrv);
        cast_slice_different_sizes::check(cx, expr, self.msrv);
        ptr_cast_constness::check_null_ptr_cast_method(cx, expr);
    }
}
