//! This is a fixture we use for tests that need lang items.
//!
//! We want to include the minimal subset of core for each test, so this file
//! supports "conditional compilation". Tests use the following syntax to include minicore:
//!
//!  //- minicore: flag1, flag2
//!
//! We then strip all the code marked with other flags.
//!
//! Available flags:
//!     add:
//!     asm:
//!     assert:
//!     as_mut: sized
//!     as_ref: sized
//!     async_fn: fn, tuple, future, copy
//!     bool_impl: option, fn
//!     builtin_impls:
//!     borrow: sized
//!     borrow_mut: borrow
//!     cell: copy, drop
//!     clone: sized
//!     coerce_pointee: derive, sized, unsize, coerce_unsized, dispatch_from_dyn
//!     coerce_unsized: unsize
//!     concat:
//!     copy: clone
//!     default: sized
//!     deref_mut: deref
//!     deref: sized
//!     derive:
//!     discriminant:
//!     drop: sized
//!     env: option
//!     eq: sized
//!     error: fmt
//!     fmt: option, result, transmute, coerce_unsized, copy, clone, derive
//!     fmt_before_1_93_0: fmt
//!     fmt_before_1_89_0: fmt_before_1_93_0
//!     fn: sized, tuple
//!     from: sized, result
//!     future: pin
//!     coroutine: pin
//!     dispatch_from_dyn: unsize, pin
//!     hash: sized
//!     include:
//!     index: sized
//!     infallible:
//!     int_impl: size_of, transmute
//!     iterator: option
//!     iterators: iterator, fn
//!     manually_drop: drop
//!     non_null:
//!     non_zero:
//!     option: panic
//!     ord: eq, option
//!     panic: fmt
//!     phantom_data:
//!     pin:
//!     pointee: copy, send, sync, ord, hash, unpin, phantom_data
//!     range:
//!     new_range:
//!     receiver: deref
//!     result:
//!     send: sized
//!     size_of: sized
//!     sized:
//!     slice:
//!     str:
//!     sync: sized
//!     transmute:
//!     try: infallible
//!     tuple:
//!     unary_ops:
//!     unpin: sized
//!     unsize: sized
//!     write: fmt
//!     todo: panic
//!     unimplemented: panic
//!     column:
//!     addr_of:
//!     offset_of:

#![rustc_coherence_is_core]
#![feature(lang_items)]

pub mod marker {
    // region:sized
    #[lang = "pointee_sized"]
    #[fundamental]
    #[rustc_specialization_trait]
    #[rustc_coinductive]
    pub trait PointeeSized {}

    #[lang = "meta_sized"]
    #[fundamental]
    #[rustc_specialization_trait]
    #[rustc_coinductive]
    pub trait MetaSized: PointeeSized {}

    #[lang = "sized"]
    #[fundamental]
    #[rustc_specialization_trait]
    #[rustc_coinductive]
    pub trait Sized: MetaSized {}
    // endregion:sized

    // region:send
    pub unsafe auto trait Send {}

    impl<T: PointeeSized> !Send for *const T {}
    impl<T: PointeeSized> !Send for *mut T {}
    // region:sync
    unsafe impl<T: Sync + PointeeSized> Send for &T {}
    unsafe impl<T: Send + PointeeSized> Send for &mut T {}
    // endregion:sync
    // endregion:send

    // region:sync
    pub unsafe auto trait Sync {}

    impl<T: PointeeSized> !Sync for *const T {}
    impl<T: PointeeSized> !Sync for *mut T {}
    // endregion:sync

    // region:unsize
    #[lang = "unsize"]
    pub trait Unsize<T: PointeeSized>: PointeeSized {}
    // endregion:unsize

    // region:unpin
    #[lang = "unpin"]
    pub auto trait Unpin {}
    // endregion:unpin

    // region:copy
    #[lang = "copy"]
    pub trait Copy: Clone {}
    // region:derive
    #[rustc_builtin_macro]
    pub macro Copy($item:item) {}
    // endregion:derive

    mod copy_impls {
        use super::{Copy, PointeeSized};

        macro_rules! impl_copy {
            ($($t:ty)*) => {
                $(
                    impl Copy for $t {}
                )*
            }
        }

        impl_copy! {
            usize u8 u16 u32 u64 u128
            isize i8 i16 i32 i64 i128
            f16 f32 f64 f128
            bool char
        }

        impl<T: PointeeSized> Copy for *const T {}
        impl<T: PointeeSized> Copy for *mut T {}
        impl<T: PointeeSized> Copy for &T {}
        impl Copy for ! {}
    }
    // endregion:copy

    // region:tuple
    #[lang = "tuple_trait"]
    pub trait Tuple {}
    // endregion:tuple

    // region:phantom_data
    #[lang = "phantom_data"]
    pub struct PhantomData<T: PointeeSized>;

    // region:clone
    impl<T: PointeeSized> Clone for PhantomData<T> {
        fn clone(&self) -> Self {
            Self
        }
    }
    // endregion:clone

    // region:copy
    impl<T: PointeeSized> Copy for PhantomData<T> {}
    // endregion:copy

    // endregion:phantom_data

    // region:discriminant
    #[lang = "discriminant_kind"]
    pub trait DiscriminantKind {
        #[lang = "discriminant_type"]
        type Discriminant;
    }
    // endregion:discriminant

    // region:coerce_pointee
    #[rustc_builtin_macro(CoercePointee, attributes(pointee))]
    #[allow_internal_unstable(dispatch_from_dyn, coerce_unsized, unsize)]
    pub macro CoercePointee($item:item) {
        /* compiler built-in */
    }
    // endregion:coerce_pointee
}

// region:default
pub mod default {
    pub trait Default: Sized {
        fn default() -> Self;
    }
    // region:derive
    #[rustc_builtin_macro(Default, attributes(default))]
    pub macro Default($item:item) {}
    // endregion:derive

    // region:builtin_impls
    macro_rules! impl_default {
        ($v:literal; $($t:ty)*) => {
            $(
                impl Default for $t {
                    fn default() -> Self {
                        $v
                    }
                }
            )*
        }
    }

    impl_default! {
        0; usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128
    }
    impl_default! {
        0.0; f16 f32 f64 f128
    }
    // endregion:builtin_impls
}
// endregion:default

// region:hash
pub mod hash {
    use crate::marker::PointeeSized;

    pub trait Hasher {}

    pub trait Hash: PointeeSized {
        fn hash<H: Hasher>(&self, state: &mut H);
    }

    // region:derive
    pub(crate) mod derive {
        #[rustc_builtin_macro]
        pub macro Hash($item:item) {}
    }
    pub use derive::Hash;
    // endregion:derive
}
// endregion:hash

// region:cell
pub mod cell {
    use crate::marker::PointeeSized;
    use crate::mem;

    #[lang = "unsafe_cell"]
    pub struct UnsafeCell<T: PointeeSized> {
        value: T,
    }

    impl<T> UnsafeCell<T> {
        pub const fn new(value: T) -> UnsafeCell<T> {
            UnsafeCell { value }
        }

        pub const fn get(&self) -> *mut T {
            self as *const UnsafeCell<T> as *const T as *mut T
        }
    }

    pub struct Cell<T: PointeeSized> {
        value: UnsafeCell<T>,
    }

    impl<T> Cell<T> {
        pub const fn new(value: T) -> Cell<T> {
            Cell { value: UnsafeCell::new(value) }
        }

        pub fn set(&self, val: T) {
            let old = self.replace(val);
            mem::drop(old);
        }

        pub fn replace(&self, val: T) -> T {
            mem::replace(unsafe { &mut *self.value.get() }, val)
        }
    }

    impl<T: Copy> Cell<T> {
        pub fn get(&self) -> T {
            unsafe { *self.value.get() }
        }
    }
}
// endregion:cell

// region:clone
pub mod clone {
    #[lang = "clone"]
    pub trait Clone: Sized {
        fn clone(&self) -> Self;
    }

    impl<T> Clone for &T {
        fn clone(&self) -> Self {
            *self
        }
    }

    // region:builtin_impls
    macro_rules! impl_clone {
        ($($t:ty)*) => {
            $(
                impl const Clone for $t {
                    fn clone(&self) -> Self {
                        *self
                    }
                }
            )*
        }
    }

    impl_clone! {
        usize u8 u16 u32 u64 u128
        isize i8 i16 i32 i64 i128
        f16 f32 f64 f128
        bool char
    }

    impl Clone for ! {
        fn clone(&self) {
            *self
        }
    }

    impl<T: Clone> Clone for [T; 0] {
        fn clone(&self) -> Self {
            []
        }
    }

    impl<T: Clone> Clone for [T; 1] {
        fn clone(&self) -> Self {
            [self[0].clone()]
        }
    }
    // endregion:builtin_impls

    // region:derive
    #[rustc_builtin_macro]
    pub macro Clone($item:item) {}
    // endregion:derive
}
// endregion:clone

pub mod convert {
    // region:from
    pub trait From<T>: Sized {
        fn from(_: T) -> Self;
    }
    pub trait Into<T>: Sized {
        fn into(self) -> T;
    }

    impl<T, U> Into<U> for T
    where
        U: From<T>,
    {
        fn into(self) -> U {
            U::from(self)
        }
    }

    impl<T> From<T> for T {
        fn from(t: T) -> T {
            t
        }
    }

    pub trait TryFrom<T>: Sized {
        type Error;
        fn try_from(value: T) -> Result<Self, Self::Error>;
    }
    pub trait TryInto<T>: Sized {
        type Error;
        fn try_into(self) -> Result<T, Self::Error>;
    }

    impl<T, U> TryInto<U> for T
    where
        U: TryFrom<T>,
    {
        type Error = U::Error;
        fn try_into(self) -> Result<U, U::Error> {
            U::try_from(self)
        }
    }
    // endregion:from

    // region:as_ref
    pub trait AsRef<T: crate::marker::PointeeSized>: crate::marker::PointeeSized {
        fn as_ref(&self) -> &T;
    }
    // endregion:as_ref
    // region:as_mut
    pub trait AsMut<T: crate::marker::PointeeSized>: crate::marker::PointeeSized {
        fn as_mut(&mut self) -> &mut T;
    }
    // endregion:as_mut
    // region:infallible
    pub enum Infallible {}
    // endregion:infallible
}

pub mod borrow {
    // region:borrow
    pub trait Borrow<Borrowed: ?Sized> {
        fn borrow(&self) -> &Borrowed;
    }
    // endregion:borrow

    // region:borrow_mut
    pub trait BorrowMut<Borrowed: ?Sized>: Borrow<Borrowed> {
        fn borrow_mut(&mut self) -> &mut Borrowed;
    }
    // endregion:borrow_mut
}

pub mod mem {
    // region:manually_drop
    use crate::marker::PointeeSized;

    #[lang = "manually_drop"]
    #[repr(transparent)]
    pub struct ManuallyDrop<T: PointeeSized> {
        value: T,
    }

    impl<T> ManuallyDrop<T> {
        pub const fn new(value: T) -> ManuallyDrop<T> {
            ManuallyDrop { value }
        }
    }

    // region:deref
    impl<T: PointeeSized> crate::ops::Deref for ManuallyDrop<T> {
        type Target = T;
        fn deref(&self) -> &T {
            &self.value
        }
    }
    // endregion:deref

    // endregion:manually_drop

    // region:drop
    pub fn drop<T>(_x: T) {}
    pub const fn replace<T>(dest: &mut T, src: T) -> T {
        unsafe {
            let result = crate::ptr::read(dest);
            crate::ptr::write(dest, src);
            result
        }
    }
    // endregion:drop

    // region:transmute
    #[rustc_intrinsic]
    pub fn transmute<Src, Dst>(src: Src) -> Dst;
    // endregion:transmute

    // region:size_of
    #[rustc_intrinsic]
    pub fn size_of<T>() -> usize;
    // endregion:size_of

    // region:discriminant
    use crate::marker::DiscriminantKind;
    pub struct Discriminant<T>(<T as DiscriminantKind>::Discriminant);
    // endregion:discriminant

    // region:offset_of
    pub macro offset_of($Container:ty, $($fields:expr)+ $(,)?) {
        // The `{}` is for better error messages
        {builtin # offset_of($Container, $($fields)+)}
    }
    // endregion:offset_of
}

pub mod ptr {
    // region:drop
    #[lang = "drop_in_place"]
    pub unsafe fn drop_in_place<T: crate::marker::PointeeSized>(to_drop: *mut T) {
        unsafe { drop_in_place(to_drop) }
    }
    pub const unsafe fn read<T>(src: *const T) -> T {
        unsafe { *src }
    }
    pub const unsafe fn write<T>(dst: *mut T, src: T) {
        unsafe {
            *dst = src;
        }
    }
    // endregion:drop

    // region:pointee
    #[lang = "pointee_trait"]
    #[rustc_deny_explicit_impl(implement_via_object = false)]
    pub trait Pointee: crate::marker::PointeeSized {
        #[lang = "metadata_type"]
        type Metadata: Copy + Send + Sync + Ord + Hash + Unpin;
    }

    #[lang = "dyn_metadata"]
    pub struct DynMetadata<Dyn: PointeeSized> {
        _phantom: crate::marker::PhantomData<Dyn>,
    }

    pub const fn metadata<T: PointeeSized>(ptr: *const T) -> <T as Pointee>::Metadata {
        loop {}
    }

    // endregion:pointee
    // region:non_null
    #[rustc_layout_scalar_valid_range_start(1)]
    #[rustc_nonnull_optimization_guaranteed]
    pub struct NonNull<T: crate::marker::PointeeSized> {
        pointer: *const T,
    }
    // region:coerce_unsized
    impl<T: crate::marker::PointeeSized, U: crate::marker::PointeeSized>
        crate::ops::CoerceUnsized<NonNull<U>> for NonNull<T>
    where
        T: crate::marker::Unsize<U>,
    {
    }
    // endregion:coerce_unsized
    // endregion:non_null

    // region:addr_of
    #[rustc_macro_transparency = "semiopaque"]
    pub macro addr_of($place:expr) {
        &raw const $place
    }
    #[rustc_macro_transparency = "semiopaque"]
    pub macro addr_of_mut($place:expr) {
        &raw mut $place
    }
    // endregion:addr_of
}

pub mod ops {
    // region:coerce_unsized
    mod unsize {
        use crate::marker::{PointeeSized, Unsize};

        #[lang = "coerce_unsized"]
        pub trait CoerceUnsized<T> {}

        impl<'a, T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<&'a mut U> for &'a mut T {}
        impl<'a, 'b: 'a, T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<&'a U> for &'b mut T {}
        impl<'a, T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<*mut U> for &'a mut T {}
        impl<'a, T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<*const U> for &'a mut T {}

        impl<'a, 'b: 'a, T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<&'a U> for &'b T {}
        impl<'a, T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<*const U> for &'a T {}

        impl<T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<*mut U> for *mut T {}
        impl<T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<*const U> for *mut T {}
        impl<T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<*const U> for *const T {}
    }
    pub use self::unsize::CoerceUnsized;
    // endregion:coerce_unsized

    // region:deref
    mod deref {
        use crate::marker::PointeeSized;

        #[lang = "deref"]
        pub trait Deref: PointeeSized {
            #[lang = "deref_target"]
            type Target: ?Sized;
            fn deref(&self) -> &Self::Target;
        }

        impl<T: PointeeSized> Deref for &T {
            type Target = T;
            fn deref(&self) -> &T {
                *self
            }
        }
        impl<T: PointeeSized> Deref for &mut T {
            type Target = T;
            fn deref(&self) -> &T {
                *self
            }
        }
        // region:deref_mut
        #[lang = "deref_mut"]
        pub trait DerefMut: Deref + PointeeSized {
            fn deref_mut(&mut self) -> &mut Self::Target;
        }
        // endregion:deref_mut

        // region:receiver
        #[lang = "receiver"]
        pub trait Receiver: PointeeSized {
            #[lang = "receiver_target"]
            type Target: ?Sized;
        }

        impl<P: PointeeSized, T: PointeeSized> Receiver for P
        where
            P: Deref<Target = T>,
        {
            type Target = T;
        }
        // endregion:receiver
    }
    pub use self::deref::{
        Deref,
        DerefMut, // :deref_mut
        Receiver, // :receiver
    };
    // endregion:deref

    // region:drop
    #[lang = "drop"]
    pub trait Drop {
        fn drop(&mut self);
    }
    // endregion:drop

    // region:index
    mod index {
        #[lang = "index"]
        pub trait Index<Idx: ?Sized> {
            type Output: ?Sized;
            fn index(&self, index: Idx) -> &Self::Output;
        }
        #[lang = "index_mut"]
        pub trait IndexMut<Idx: ?Sized>: Index<Idx> {
            fn index_mut(&mut self, index: Idx) -> &mut Self::Output;
        }

        // region:slice
        impl<T, I> Index<I> for [T]
        where
            I: SliceIndex<[T]>,
        {
            type Output = I::Output;
            fn index(&self, _index: I) -> &I::Output {
                loop {}
            }
        }
        impl<T, I> IndexMut<I> for [T]
        where
            I: SliceIndex<[T]>,
        {
            fn index_mut(&mut self, _index: I) -> &mut I::Output {
                loop {}
            }
        }

        impl<T, I, const N: usize> Index<I> for [T; N]
        where
            I: SliceIndex<[T]>,
        {
            type Output = I::Output;
            fn index(&self, _index: I) -> &I::Output {
                loop {}
            }
        }
        impl<T, I, const N: usize> IndexMut<I> for [T; N]
        where
            I: SliceIndex<[T]>,
        {
            fn index_mut(&mut self, _index: I) -> &mut I::Output {
                loop {}
            }
        }

        pub unsafe trait SliceIndex<T: ?Sized> {
            type Output: ?Sized;
        }
        unsafe impl<T> SliceIndex<[T]> for usize {
            type Output = T;
        }
        // endregion:slice
    }
    pub use self::index::{Index, IndexMut};
    // endregion:index

    // region:range
    mod range {
        #[lang = "RangeFull"]
        pub struct RangeFull;

        #[lang = "Range"]
        pub struct Range<Idx> {
            pub start: Idx,
            pub end: Idx,
        }

        #[lang = "RangeFrom"]
        pub struct RangeFrom<Idx> {
            pub start: Idx,
        }

        #[lang = "RangeTo"]
        pub struct RangeTo<Idx> {
            pub end: Idx,
        }

        #[lang = "RangeInclusive"]
        pub struct RangeInclusive<Idx> {
            pub(crate) start: Idx,
            pub(crate) end: Idx,
            pub(crate) exhausted: bool,
        }

        #[lang = "RangeToInclusive"]
        pub struct RangeToInclusive<Idx> {
            pub end: Idx,
        }
    }
    pub use self::range::{Range, RangeFrom, RangeFull, RangeTo};
    pub use self::range::{RangeInclusive, RangeToInclusive};
    // endregion:range

    // region:fn
    mod function {
        use crate::marker::Tuple;

        #[lang = "fn"]
        #[fundamental]
        #[rustc_paren_sugar]
        pub trait Fn<Args: Tuple>: FnMut<Args> {
            extern "rust-call" fn call(&self, args: Args) -> Self::Output;
        }

        #[lang = "fn_mut"]
        #[fundamental]
        #[rustc_paren_sugar]
        pub trait FnMut<Args: Tuple>: FnOnce<Args> {
            extern "rust-call" fn call_mut(&mut self, args: Args) -> Self::Output;
        }

        #[lang = "fn_once"]
        #[fundamental]
        #[rustc_paren_sugar]
        pub trait FnOnce<Args: Tuple> {
            #[lang = "fn_once_output"]
            type Output;
            extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
        }

        mod impls {
            use crate::marker::Tuple;

            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_unstable(feature = "const_fn_trait_ref_impls", issue = "101803")]
            impl<A: Tuple, F: ?Sized> const Fn<A> for &F
            where
                F: [const] Fn<A>,
            {
                extern "rust-call" fn call(&self, args: A) -> F::Output {
                    (**self).call(args)
                }
            }

            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_unstable(feature = "const_fn_trait_ref_impls", issue = "101803")]
            impl<A: Tuple, F: ?Sized> const FnMut<A> for &F
            where
                F: [const] Fn<A>,
            {
                extern "rust-call" fn call_mut(&mut self, args: A) -> F::Output {
                    (**self).call(args)
                }
            }

            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_unstable(feature = "const_fn_trait_ref_impls", issue = "101803")]
            impl<A: Tuple, F: ?Sized> const FnOnce<A> for &F
            where
                F: [const] Fn<A>,
            {
                type Output = F::Output;

                extern "rust-call" fn call_once(self, args: A) -> F::Output {
                    (*self).call(args)
                }
            }

            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_unstable(feature = "const_fn_trait_ref_impls", issue = "101803")]
            impl<A: Tuple, F: ?Sized> const FnMut<A> for &mut F
            where
                F: [const] FnMut<A>,
            {
                extern "rust-call" fn call_mut(&mut self, args: A) -> F::Output {
                    (*self).call_mut(args)
                }
            }

            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_unstable(feature = "const_fn_trait_ref_impls", issue = "101803")]
            impl<A: Tuple, F: ?Sized> const FnOnce<A> for &mut F
            where
                F: [const] FnMut<A>,
            {
                type Output = F::Output;
                extern "rust-call" fn call_once(self, args: A) -> F::Output {
                    (*self).call_mut(args)
                }
            }
        }
    }
    pub use self::function::{Fn, FnMut, FnOnce};
    // endregion:fn

    // region:async_fn
    mod async_function {
        use crate::{future::Future, marker::Tuple};

        #[lang = "async_fn"]
        #[fundamental]
        #[rustc_paren_sugar]
        pub trait AsyncFn<Args: Tuple>: AsyncFnMut<Args> {
            extern "rust-call" fn async_call(&self, args: Args) -> Self::CallRefFuture<'_>;
        }

        #[lang = "async_fn_mut"]
        #[fundamental]
        #[rustc_paren_sugar]
        pub trait AsyncFnMut<Args: Tuple>: AsyncFnOnce<Args> {
            #[lang = "call_ref_future"]
            type CallRefFuture<'a>: Future<Output = Self::Output>
            where
                Self: 'a;
            extern "rust-call" fn async_call_mut(&mut self, args: Args) -> Self::CallRefFuture<'_>;
        }

        #[lang = "async_fn_once"]
        #[fundamental]
        #[rustc_paren_sugar]
        pub trait AsyncFnOnce<Args: Tuple> {
            #[lang = "async_fn_once_output"]
            type Output;
            #[lang = "call_once_future"]
            type CallOnceFuture: Future<Output = Self::Output>;
            extern "rust-call" fn async_call_once(self, args: Args) -> Self::CallOnceFuture;
        }

        mod impls {
            use super::{AsyncFn, AsyncFnMut, AsyncFnOnce};
            use crate::marker::Tuple;

            impl<A: Tuple, F: ?Sized> AsyncFn<A> for &F
            where
                F: AsyncFn<A>,
            {
                extern "rust-call" fn async_call(&self, args: A) -> Self::CallRefFuture<'_> {
                    F::async_call(*self, args)
                }
            }

            impl<A: Tuple, F: ?Sized> AsyncFnMut<A> for &F
            where
                F: AsyncFn<A>,
            {
                type CallRefFuture<'a>
                    = F::CallRefFuture<'a>
                where
                    Self: 'a;

                extern "rust-call" fn async_call_mut(
                    &mut self,
                    args: A,
                ) -> Self::CallRefFuture<'_> {
                    F::async_call(*self, args)
                }
            }

            impl<'a, A: Tuple, F: ?Sized> AsyncFnOnce<A> for &'a F
            where
                F: AsyncFn<A>,
            {
                type Output = F::Output;
                type CallOnceFuture = F::CallRefFuture<'a>;

                extern "rust-call" fn async_call_once(self, args: A) -> Self::CallOnceFuture {
                    F::async_call(self, args)
                }
            }

            impl<A: Tuple, F: ?Sized> AsyncFnMut<A> for &mut F
            where
                F: AsyncFnMut<A>,
            {
                type CallRefFuture<'a>
                    = F::CallRefFuture<'a>
                where
                    Self: 'a;

                extern "rust-call" fn async_call_mut(
                    &mut self,
                    args: A,
                ) -> Self::CallRefFuture<'_> {
                    F::async_call_mut(*self, args)
                }
            }

            impl<'a, A: Tuple, F: ?Sized> AsyncFnOnce<A> for &'a mut F
            where
                F: AsyncFnMut<A>,
            {
                type Output = F::Output;
                type CallOnceFuture = F::CallRefFuture<'a>;

                extern "rust-call" fn async_call_once(self, args: A) -> Self::CallOnceFuture {
                    F::async_call_mut(self, args)
                }
            }
        }
    }
    pub use self::async_function::{AsyncFn, AsyncFnMut, AsyncFnOnce};
    // endregion:async_fn

    // region:try
    mod try_ {
        use crate::convert::Infallible;

        pub enum ControlFlow<B, C = ()> {
            #[lang = "Continue"]
            Continue(C),
            #[lang = "Break"]
            Break(B),
        }
        pub trait FromResidual<R = <Self as Try>::Residual> {
            #[lang = "from_residual"]
            fn from_residual(residual: R) -> Self;
        }
        #[lang = "Try"]
        pub trait Try: FromResidual<Self::Residual> {
            type Output;
            type Residual;
            #[lang = "from_output"]
            fn from_output(output: Self::Output) -> Self;
            #[lang = "branch"]
            fn branch(self) -> ControlFlow<Self::Residual, Self::Output>;
        }

        impl<B, C> Try for ControlFlow<B, C> {
            type Output = C;
            type Residual = ControlFlow<B, Infallible>;
            fn from_output(output: Self::Output) -> Self {
                ControlFlow::Continue(output)
            }
            fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
                match self {
                    ControlFlow::Continue(x) => ControlFlow::Continue(x),
                    ControlFlow::Break(x) => ControlFlow::Break(ControlFlow::Break(x)),
                }
            }
        }

        impl<B, C> FromResidual for ControlFlow<B, C> {
            fn from_residual(residual: ControlFlow<B, Infallible>) -> Self {
                match residual {
                    ControlFlow::Break(b) => ControlFlow::Break(b),
                    ControlFlow::Continue(_) => loop {},
                }
            }
        }
        // region:option
        impl<T> Try for Option<T> {
            type Output = T;
            type Residual = Option<Infallible>;
            fn from_output(output: Self::Output) -> Self {
                Some(output)
            }
            fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
                match self {
                    Some(x) => ControlFlow::Continue(x),
                    None => ControlFlow::Break(None),
                }
            }
        }

        impl<T> FromResidual for Option<T> {
            fn from_residual(x: Option<Infallible>) -> Self {
                match x {
                    None => None,
                    Some(_) => loop {},
                }
            }
        }
        // endregion:option
        // region:result
        // region:from
        use crate::convert::From;

        impl<T, E> Try for Result<T, E> {
            type Output = T;
            type Residual = Result<Infallible, E>;

            fn from_output(output: Self::Output) -> Self {
                Ok(output)
            }

            fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
                match self {
                    Ok(v) => ControlFlow::Continue(v),
                    Err(e) => ControlFlow::Break(Err(e)),
                }
            }
        }

        impl<T, E, F: From<E>> FromResidual<Result<Infallible, E>> for Result<T, F> {
            fn from_residual(residual: Result<Infallible, E>) -> Self {
                match residual {
                    Err(e) => Err(F::from(e)),
                    Ok(_) => loop {},
                }
            }
        }
        // endregion:from
        // endregion:result
    }
    pub use self::try_::{ControlFlow, FromResidual, Try};
    // endregion:try

    // region:add
    #[lang = "add"]
    pub trait Add<Rhs = Self> {
        type Output;
        fn add(self, rhs: Rhs) -> Self::Output;
    }

    #[lang = "add_assign"]
    pub const trait AddAssign<Rhs = Self> {
        fn add_assign(&mut self, rhs: Rhs);
    }

    // region:builtin_impls
    macro_rules! add_impl {
        ($($t:ty)*) => ($(
            impl const Add for $t {
                type Output = $t;
                fn add(self, other: $t) -> $t { self + other }
            }
            impl AddAssign for $t {
                fn add_assign(&mut self, other: $t) { *self += other; }
            }
        )*)
    }

    add_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f16 f32 f64 f128 }
    // endregion:builtin_impls
    // endregion:add

    // region:unary_ops
    #[lang = "not"]
    pub const trait Not {
        type Output;

        #[must_use]
        fn not(self) -> Self::Output;
    }

    #[lang = "neg"]
    pub const trait Neg {
        type Output;

        #[must_use = "this returns the result of the operation, without modifying the original"]
        fn neg(self) -> Self::Output;
    }
    // endregion:unary_ops

    // region:coroutine
    mod coroutine {
        use crate::pin::Pin;

        #[lang = "coroutine"]
        pub trait Coroutine<R = ()> {
            #[lang = "coroutine_yield"]
            type Yield;
            #[lang = "coroutine_return"]
            type Return;
            fn resume(self: Pin<&mut Self>, arg: R) -> CoroutineState<Self::Yield, Self::Return>;
        }

        #[lang = "coroutine_state"]
        pub enum CoroutineState<Y, R> {
            Yielded(Y),
            Complete(R),
        }
    }
    pub use self::coroutine::{Coroutine, CoroutineState};
    // endregion:coroutine

    // region:dispatch_from_dyn
    mod dispatch_from_dyn {
        use crate::marker::{PointeeSized, Unsize};

        #[lang = "dispatch_from_dyn"]
        pub trait DispatchFromDyn<T> {}

        impl<'a, T: PointeeSized + Unsize<U>, U: PointeeSized> DispatchFromDyn<&'a U> for &'a T {}

        impl<'a, T: PointeeSized + Unsize<U>, U: PointeeSized> DispatchFromDyn<&'a mut U> for &'a mut T {}

        impl<T: PointeeSized + Unsize<U>, U: PointeeSized> DispatchFromDyn<*const U> for *const T {}

        impl<T: PointeeSized + Unsize<U>, U: PointeeSized> DispatchFromDyn<*mut U> for *mut T {}
    }
    pub use self::dispatch_from_dyn::DispatchFromDyn;
    // endregion:dispatch_from_dyn
}

// region:new_range
pub mod range {
    #[lang = "RangeCopy"]
    pub struct Range<Idx> {
        pub start: Idx,
        pub end: Idx,
    }

    #[lang = "RangeFromCopy"]
    pub struct RangeFrom<Idx> {
        pub start: Idx,
    }

    #[lang = "RangeInclusiveCopy"]
    pub struct RangeInclusive<Idx> {
        pub start: Idx,
        pub end: Idx,
    }

    #[lang = "RangeToInclusiveCopy"]
    pub struct RangeToInclusive<Idx> {
        pub end: Idx,
    }
}
// endregion:new_range

// region:eq
pub mod cmp {
    use crate::marker::PointeeSized;

    #[lang = "eq"]
    pub trait PartialEq<Rhs: PointeeSized = Self>: PointeeSized {
        fn eq(&self, other: &Rhs) -> bool;
        fn ne(&self, other: &Rhs) -> bool {
            !self.eq(other)
        }
    }

    pub trait Eq: PartialEq<Self> + PointeeSized {}

    // region:builtin_impls
    impl PartialEq for () {
        fn eq(&self, other: &()) -> bool {
            true
        }
    }
    // endregion:builtin_impls

    // region:derive
    #[rustc_builtin_macro]
    pub macro PartialEq($item:item) {}
    #[rustc_builtin_macro]
    pub macro Eq($item:item) {}
    // endregion:derive

    // region:ord
    #[lang = "partial_ord"]
    pub trait PartialOrd<Rhs: PointeeSized = Self>: PartialEq<Rhs> + PointeeSized {
        fn partial_cmp(&self, other: &Rhs) -> Option<Ordering>;
    }

    pub trait Ord: Eq + PartialOrd<Self> + PointeeSized {
        fn cmp(&self, other: &Self) -> Ordering;
    }

    pub enum Ordering {
        Less = -1,
        Equal = 0,
        Greater = 1,
    }

    // region:derive
    #[rustc_builtin_macro]
    pub macro PartialOrd($item:item) {}
    #[rustc_builtin_macro]
    pub macro Ord($item:item) {}
    // endregion:derive

    // endregion:ord
}
// endregion:eq

// region:fmt
pub mod fmt {
    use crate::marker::PointeeSized;

    pub struct Error;
    pub type Result = crate::result::Result<(), Error>;
    pub struct Formatter<'a>(&'a ());
    pub struct DebugTuple;
    pub struct DebugStruct;
    impl Formatter<'_> {
        pub fn debug_tuple(&mut self, _name: &str) -> DebugTuple {
            DebugTuple
        }

        pub fn debug_struct(&mut self, _name: &str) -> DebugStruct {
            DebugStruct
        }
    }

    impl DebugTuple {
        pub fn field(&mut self, _value: &dyn Debug) -> &mut Self {
            self
        }

        pub fn finish(&mut self) -> Result {
            Ok(())
        }
    }

    impl DebugStruct {
        pub fn field(&mut self, _name: &str, _value: &dyn Debug) -> &mut Self {
            self
        }

        pub fn finish(&mut self) -> Result {
            Ok(())
        }
    }

    pub trait Debug: PointeeSized {
        fn fmt(&self, f: &mut Formatter<'_>) -> Result;
    }
    pub trait Display: PointeeSized {
        fn fmt(&self, f: &mut Formatter<'_>) -> Result;
    }

    mod rt {
        use super::*;

        extern "C" {
            type Opaque;
        }

        #[derive(Copy, Clone)]
        #[lang = "format_argument"]
        pub struct Argument<'a> {
            value: &'a Opaque,
            formatter: fn(&Opaque, &mut Formatter<'_>) -> Result,
        }

        impl<'a> Argument<'a> {
            pub fn new<'b, T>(x: &'b T, f: fn(&T, &mut Formatter<'_>) -> Result) -> Argument<'b> {
                use crate::mem::transmute;
                unsafe { Argument { formatter: transmute(f), value: transmute(x) } }
            }

            pub fn new_display<'b, T: crate::fmt::Display>(x: &'b T) -> Argument<'_> {
                Self::new(x, crate::fmt::Display::fmt)
            }
        }

        #[lang = "format_alignment"]
        pub enum Alignment {
            Left,
            Right,
            Center,
            Unknown,
        }

        // region:fmt_before_1_93_0
        #[lang = "format_count"]
        pub enum Count {
            Is(usize),
            Param(usize),
            Implied,
        }

        #[lang = "format_placeholder"]
        pub struct Placeholder {
            pub position: usize,
            pub fill: char,
            pub align: Alignment,
            pub flags: u32,
            pub precision: Count,
            pub width: Count,
        }

        impl Placeholder {
            pub const fn new(
                position: usize,
                fill: char,
                align: Alignment,
                flags: u32,
                precision: Count,
                width: Count,
            ) -> Self {
                Placeholder { position, fill, align, flags, precision, width }
            }
        }
        // endregion:fmt_before_1_93_0

        // region:fmt_before_1_89_0
        #[lang = "format_unsafe_arg"]
        pub struct UnsafeArg {
            _private: (),
        }

        impl UnsafeArg {
            pub unsafe fn new() -> Self {
                UnsafeArg { _private: () }
            }
        }
        // endregion:fmt_before_1_89_0
    }

    // region:fmt_before_1_93_0
    #[derive(Copy, Clone)]
    #[lang = "format_arguments"]
    pub struct Arguments<'a> {
        pieces: &'a [&'static str],
        fmt: Option<&'a [rt::Placeholder]>,
        args: &'a [rt::Argument<'a>],
    }

    impl<'a> Arguments<'a> {
        pub const fn new_v1(pieces: &'a [&'static str], args: &'a [Argument<'a>]) -> Arguments<'a> {
            Arguments { pieces, fmt: None, args }
        }

        pub const fn new_const(pieces: &'a [&'static str]) -> Arguments<'a> {
            Arguments { pieces, fmt: None, args: &[] }
        }

        // region:fmt_before_1_89_0
        pub fn new_v1_formatted(
            pieces: &'a [&'static str],
            args: &'a [rt::Argument<'a>],
            fmt: &'a [rt::Placeholder],
            _unsafe_arg: rt::UnsafeArg,
        ) -> Arguments<'a> {
            Arguments { pieces, fmt: Some(fmt), args }
        }
        // endregion:fmt_before_1_89_0

        // region:!fmt_before_1_89_0
        pub unsafe fn new_v1_formatted(
            pieces: &'a [&'static str],
            args: &'a [rt::Argument<'a>],
            fmt: &'a [rt::Placeholder],
        ) -> Arguments<'a> {
            Arguments { pieces, fmt: Some(fmt), args }
        }
        // endregion:!fmt_before_1_89_0

        pub fn from_str_nonconst(s: &'static str) -> Arguments<'a> {
            Self::from_str(s)
        }

        pub const fn from_str(s: &'static str) -> Arguments<'a> {
            Arguments { pieces: &[s], fmt: None, args: &[] }
        }

        pub const fn as_str(&self) -> Option<&'static str> {
            match (self.pieces, self.args) {
                ([], []) => Some(""),
                ([s], []) => Some(s),
                _ => None,
            }
        }
    }
    // endregion:fmt_before_1_93_0

    // region:!fmt_before_1_93_0
    #[lang = "format_arguments"]
    #[derive(Copy, Clone)]
    pub struct Arguments<'a> {
        // This is a non-faithful representation of `core::fmt::Arguments`, because the real one
        // is too complex for minicore.
        message: Option<&'a str>,
    }

    impl<'a> Arguments<'a> {
        pub unsafe fn new<const N: usize, const M: usize>(
            _template: &'a [u8; N],
            _args: &'a [rt::Argument<'a>; M],
        ) -> Arguments<'a> {
            Arguments { message: None }
        }

        pub fn from_str_nonconst(s: &'static str) -> Arguments<'a> {
            Arguments { message: Some(s) }
        }

        pub const fn from_str(s: &'static str) -> Arguments<'a> {
            Arguments { message: Some(s) }
        }

        pub fn as_str(&self) -> Option<&'static str> {
            match self.message {
                Some(s) => unsafe { Some(&*(s as *const str)) },
                None => None,
            }
        }
    }
    // endregion:!fmt_before_1_93_0

    // region:derive
    pub(crate) mod derive {
        #[rustc_builtin_macro]
        pub macro Debug($item:item) {}
    }
    pub use derive::Debug;
    // endregion:derive

    // region:builtin_impls
    macro_rules! impl_debug {
        ($($t:ty)*) => {
            $(
                impl const Debug for $t {
                    fn fmt(&self, _f: &mut Formatter<'_>) -> Result {
                        Ok(())
                    }
                }
            )*
        }
    }

    impl_debug! {
        usize u8 u16 u32 u64 u128
        isize i8 i16 i32 i64 i128
        f16 f32 f64 f128
        bool char
    }

    impl<T: Debug> Debug for [T] {
        fn fmt(&self, _f: &mut Formatter<'_>) -> Result {
            Ok(())
        }
    }

    impl<T: Debug + PointeeSized> Debug for &T {
        fn fmt(&self, f: &mut Formatter<'_>) -> Result {
            (&**self).fmt(f)
        }
    }
    // endregion:builtin_impls
}
// endregion:fmt

// region:slice
pub mod slice {
    #[lang = "slice"]
    impl<T> [T] {
        #[lang = "slice_len_fn"]
        pub fn len(&self) -> usize {
            loop {}
        }
    }
}
// endregion:slice

// region:option
pub mod option {
    pub enum Option<T> {
        #[lang = "None"]
        None,
        #[lang = "Some"]
        Some(T),
    }

    // region:copy
    impl<T: Copy> Copy for Option<T> {}
    // endregion:copy

    impl<T> Option<T> {
        pub const fn unwrap(self) -> T {
            match self {
                Some(val) => val,
                None => panic!("called `Option::unwrap()` on a `None` value"),
            }
        }

        pub const fn as_ref(&self) -> Option<&T> {
            match self {
                Some(x) => Some(x),
                None => None,
            }
        }

        pub fn and<U>(self, _optb: Option<U>) -> Option<U> {
            loop {}
        }
        pub fn unwrap_or(self, default: T) -> T {
            match self {
                Some(val) => val,
                None => default,
            }
        }
        // region:result
        pub const fn ok_or<E>(self, err: E) -> Result<T, E> {
            match self {
                Some(v) => Ok(v),
                None => Err(err),
            }
        }
        // endregion:result
        // region:fn
        pub fn and_then<U, F>(self, _f: F) -> Option<U>
        where
            F: FnOnce(T) -> Option<U>,
        {
            loop {}
        }
        pub fn unwrap_or_else<F>(self, _f: F) -> T
        where
            F: FnOnce() -> T,
        {
            loop {}
        }
        pub fn map_or<U, F>(self, _default: U, _f: F) -> U
        where
            F: FnOnce(T) -> U,
        {
            loop {}
        }
        pub fn map_or_else<U, D, F>(self, _default: D, _f: F) -> U
        where
            D: FnOnce() -> U,
            F: FnOnce(T) -> U,
        {
            loop {}
        }
        // endregion:fn
    }
}
// endregion:option

// region:result
pub mod result {
    pub enum Result<T, E> {
        #[lang = "Ok"]
        Ok(T),
        #[lang = "Err"]
        Err(E),
    }
}
// endregion:result

// region:pin
pub mod pin {
    #[lang = "pin"]
    #[fundamental]
    pub struct Pin<P> {
        pointer: P,
    }
    impl<P> Pin<P> {
        pub fn new(pointer: P) -> Pin<P> {
            Pin { pointer }
        }
    }
    // region:deref
    impl<P: crate::ops::Deref> crate::ops::Deref for Pin<P> {
        type Target = P::Target;
        fn deref(&self) -> &P::Target {
            loop {}
        }
    }
    // endregion:deref
    // region:dispatch_from_dyn
    impl<Ptr, U> crate::ops::DispatchFromDyn<Pin<U>> for Pin<Ptr> where
        Ptr: crate::ops::DispatchFromDyn<U>
    {
    }
    // endregion:dispatch_from_dyn
    // region:coerce_unsized
    impl<Ptr, U> crate::ops::CoerceUnsized<Pin<U>> for Pin<Ptr> where Ptr: crate::ops::CoerceUnsized<U> {}
    // endregion:coerce_unsized
}
// endregion:pin

// region:future
pub mod future {
    use crate::{
        pin::Pin,
        task::{Context, Poll},
    };

    #[doc(notable_trait)]
    #[lang = "future_trait"]
    pub trait Future {
        #[lang = "future_output"]
        type Output;
        #[lang = "poll"]
        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
    }

    pub trait IntoFuture {
        type Output;
        type IntoFuture: Future<Output = Self::Output>;
        #[lang = "into_future"]
        fn into_future(self) -> Self::IntoFuture;
    }

    impl<F: Future> IntoFuture for F {
        type Output = F::Output;
        type IntoFuture = F;
        fn into_future(self) -> F {
            self
        }
    }
}
pub mod task {
    pub enum Poll<T> {
        #[lang = "Ready"]
        Ready(T),
        #[lang = "Pending"]
        Pending,
    }

    pub struct Context<'a> {
        waker: &'a (),
    }
}
// endregion:future

// region:iterator
pub mod iter {
    // region:iterators
    mod adapters {
        pub struct Take<I> {
            iter: I,
            n: usize,
        }
        impl<I> Iterator for Take<I>
        where
            I: Iterator,
        {
            type Item = <I as Iterator>::Item;

            fn next(&mut self) -> Option<<I as Iterator>::Item> {
                loop {}
            }
        }

        pub struct FilterMap<I, F> {
            iter: I,
            f: F,
        }
        impl<B, I: Iterator, F> Iterator for FilterMap<I, F>
        where
            F: FnMut(I::Item) -> Option<B>,
        {
            type Item = B;

            #[inline]
            fn next(&mut self) -> Option<B> {
                loop {}
            }
        }
    }
    pub use self::adapters::{FilterMap, Take};

    mod sources {
        mod repeat {
            pub fn repeat<T>(_elt: T) -> Repeat<T> {
                loop {}
            }

            pub struct Repeat<A> {
                element: A,
            }

            impl<A> Iterator for Repeat<A> {
                type Item = A;

                fn next(&mut self) -> Option<A> {
                    loop {}
                }
            }
        }
        pub use self::repeat::{Repeat, repeat};
    }
    pub use self::sources::{Repeat, repeat};
    // endregion:iterators

    mod traits {
        mod iterator {
            use crate::marker::PointeeSized;

            #[doc(notable_trait)]
            #[lang = "iterator"]
            pub trait Iterator {
                type Item;
                #[lang = "next"]
                fn next(&mut self) -> Option<Self::Item>;
                fn nth(&mut self, n: usize) -> Option<Self::Item> {
                    loop {}
                }
                fn by_ref(&mut self) -> &mut Self
                where
                    Self: Sized,
                {
                    self
                }
                // region:iterators
                fn take(self, n: usize) -> crate::iter::Take<Self>
                where
                    Self: Sized,
                {
                    loop {}
                }
                fn filter_map<B, F>(self, _f: F) -> crate::iter::FilterMap<Self, F>
                where
                    Self: Sized,
                    F: FnMut(Self::Item) -> Option<B>,
                {
                    loop {}
                }
                fn collect<B: FromIterator<Self::Item>>(self) -> B
                where
                    Self: Sized,
                {
                    loop {}
                }
                // endregion:iterators
            }
            impl<I: Iterator + PointeeSized> Iterator for &mut I {
                type Item = I::Item;
                fn next(&mut self) -> Option<I::Item> {
                    (**self).next()
                }
            }
        }
        pub use self::iterator::Iterator;

        mod collect {
            pub trait IntoIterator {
                type Item;
                type IntoIter: Iterator<Item = Self::Item>;
                #[lang = "into_iter"]
                fn into_iter(self) -> Self::IntoIter;
            }
            impl<I: Iterator> IntoIterator for I {
                type Item = I::Item;
                type IntoIter = I;
                fn into_iter(self) -> I {
                    self
                }
            }
            struct IndexRange {
                start: usize,
                end: usize,
            }
            pub struct IntoIter<T, const N: usize> {
                data: [T; N],
                range: IndexRange,
            }
            impl<T, const N: usize> IntoIterator for [T; N] {
                type Item = T;
                type IntoIter = IntoIter<T, N>;
                fn into_iter(self) -> Self::IntoIter {
                    IntoIter { data: self, range: IndexRange { start: 0, end: loop {} } }
                }
            }
            impl<T, const N: usize> Iterator for IntoIter<T, N> {
                type Item = T;
                fn next(&mut self) -> Option<T> {
                    loop {}
                }
            }
            pub struct Iter<'a, T> {
                slice: &'a [T],
            }
            impl<'a, T> IntoIterator for &'a [T; N] {
                type Item = &'a T;
                type IntoIter = Iter<'a, T>;
                fn into_iter(self) -> Self::IntoIter {
                    loop {}
                }
            }
            impl<'a, T> IntoIterator for &'a [T] {
                type Item = &'a T;
                type IntoIter = Iter<'a, T>;
                fn into_iter(self) -> Self::IntoIter {
                    loop {}
                }
            }
            impl<'a, T> Iterator for Iter<'a, T> {
                type Item = &'a T;
                fn next(&mut self) -> Option<T> {
                    loop {}
                }
            }
            pub trait FromIterator<A>: Sized {
                fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self;
            }
        }
        pub use self::collect::{FromIterator, IntoIterator};
    }
    pub use self::traits::{FromIterator, IntoIterator, Iterator};
}
// endregion:iterator

// region:str
pub mod str {
    pub const unsafe fn from_utf8_unchecked(v: &[u8]) -> &str {
        ""
    }
    pub trait FromStr: Sized {
        type Err;
        fn from_str(s: &str) -> Result<Self, Self::Err>;
    }
    impl str {
        pub fn parse<F: FromStr>(&self) -> Result<F, F::Err> {
            FromStr::from_str(self)
        }
    }
}
// endregion:str

// region:panic
mod panic {
    pub macro panic_2021 {
        () => ({
            const fn panic_cold_explicit() -> ! {
                $crate::panicking::panic_explicit()
            }
            panic_cold_explicit();
        }),
        // Special-case the single-argument case for const_panic.
        ("{}", $arg:expr $(,)?) => ({
            #[rustc_const_panic_str] // enforce a &&str argument in const-check and hook this by const-eval
            #[rustc_do_not_const_check] // hooked by const-eval
            const fn panic_cold_display<T: $crate::fmt::Display>(arg: &T) -> ! {
                $crate::panicking::panic_display(arg)
            }
            panic_cold_display(&$arg);
        }),
        ($($t:tt)+) => ({
            // Semicolon to prevent temporaries inside the formatting machinery from
            // being considered alive in the caller after the panic_fmt call.
            $crate::panicking::panic_fmt($crate::const_format_args!($($t)+));
        }),
    }
}

mod panicking {
    #[rustc_const_panic_str] // enforce a &&str argument in const-check and hook this by const-eval
    pub const fn panic_display<T: crate::fmt::Display>(x: &T) -> ! {
        panic_fmt(crate::format_args!("{}", *x));
    }

    // This function is used instead of panic_fmt in const eval.
    #[lang = "const_panic_fmt"]
    pub const fn const_panic_fmt(fmt: crate::fmt::Arguments<'_>) -> ! {
        if let Some(msg) = fmt.as_str() {
            // The panic_display function is hooked by const eval.
            panic_display(&msg);
        } else {
            loop {}
        }
    }

    #[lang = "panic_fmt"] // needed for const-evaluated panics
    pub const fn panic_fmt(fmt: crate::fmt::Arguments<'_>) -> ! {
        loop {}
    }

    #[lang = "panic"]
    pub const fn panic(expr: &'static str) -> ! {
        panic_fmt(crate::fmt::Arguments::from_str(expr))
    }
}
// endregion:panic

// region:asm
mod arch {
    #[rustc_builtin_macro]
    pub macro asm("assembly template", $(operands,)* $(options($(option),*))?) {
        /* compiler built-in */
    }
    #[rustc_builtin_macro]
    pub macro global_asm("assembly template", $(operands,)* $(options($(option),*))?) {
        /* compiler built-in */
    }
    #[rustc_builtin_macro]
    pub macro naked_asm("assembly template", $(operands,)* $(options($(option),*))?) {
        /* compiler built-in */
    }
}
// endregion:asm

#[macro_use]
mod macros {
    // region:panic
    #[macro_export]
    #[rustc_builtin_macro(core_panic)]
    macro_rules! panic {
        ($($arg:tt)*) => {
            /* compiler built-in */
        };
    }
    // endregion:panic

    // region:write
    #[macro_export]
    macro_rules! write {
        ($dst:expr, $($arg:tt)*) => {
            $dst.write_fmt($crate::format_args!($($arg)*))
        };
    }

    #[macro_export]
    #[allow_internal_unstable(format_args_nl)]
    macro_rules! writeln {
        ($dst:expr $(,)?) => {
            $crate::write!($dst, "\n")
        };
        ($dst:expr, $($arg:tt)*) => {
            $dst.write_fmt($crate::format_args_nl!($($arg)*))
        };
    }
    // endregion:write

    // region:assert
    #[macro_export]
    #[rustc_builtin_macro]
    #[allow_internal_unstable(core_panic, edition_panic, generic_assert_internals)]
    macro_rules! assert {
        ($($arg:tt)*) => {
            /* compiler built-in */
        };
    }
    // endregion:assert

    // region:fmt
    #[allow_internal_unstable(fmt_internals, const_fmt_arguments_new)]
    #[macro_export]
    #[rustc_builtin_macro]
    macro_rules! const_format_args {
        ($fmt:expr) => {{ /* compiler built-in */ }};
        ($fmt:expr, $($args:tt)*) => {{ /* compiler built-in */ }};
    }

    #[allow_internal_unstable(fmt_internals)]
    #[macro_export]
    #[rustc_builtin_macro]
    macro_rules! format_args {
        ($fmt:expr) => {{ /* compiler built-in */ }};
        ($fmt:expr, $($args:tt)*) => {{ /* compiler built-in */ }};
    }

    #[allow_internal_unstable(fmt_internals)]
    #[macro_export]
    #[rustc_builtin_macro]
    macro_rules! format_args_nl {
        ($fmt:expr) => {{ /* compiler built-in */ }};
        ($fmt:expr, $($args:tt)*) => {{ /* compiler built-in */ }};
    }

    #[macro_export]
    macro_rules! print {
        ($($arg:tt)*) => {{
            $crate::io::_print($crate::format_args!($($arg)*));
        }};
    }

    // endregion:fmt

    // region:todo
    #[macro_export]
    #[allow_internal_unstable(core_panic)]
    macro_rules! todo {
        () => {
            $crate::panicking::panic("not yet implemented")
        };
        ($($arg:tt)+) => {
            $crate::panic!("not yet implemented: {}", $crate::format_args!($($arg)+))
        };
    }
    // endregion:todo

    // region:unimplemented
    #[macro_export]
    #[allow_internal_unstable(core_panic)]
    macro_rules! unimplemented {
        () => {
            $crate::panicking::panic("not implemented")
        };
        ($($arg:tt)+) => {
            $crate::panic!("not implemented: {}", $crate::format_args!($($arg)+))
        };
    }
    // endregion:unimplemented

    // region:derive
    pub(crate) mod builtin {
        #[rustc_builtin_macro]
        pub macro derive($item:item) {
            /* compiler built-in */
        }

        #[rustc_builtin_macro]
        pub macro derive_const($item:item) {
            /* compiler built-in */
        }
    }
    // endregion:derive

    // region:include
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! include {
        ($file:expr $(,)?) => {{ /* compiler built-in */ }};
    }
    // endregion:include

    // region:concat
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! concat {}
    // endregion:concat

    // region:env
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! env {}
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! option_env {}
    // endregion:env
}

// region:non_zero
pub mod num {
    #[repr(transparent)]
    #[rustc_layout_scalar_valid_range_start(1)]
    #[rustc_nonnull_optimization_guaranteed]
    pub struct NonZeroU8(u8);
}
// endregion:non_zero

// region:bool_impl
#[lang = "bool"]
impl bool {
    pub fn then_some<T>(self, t: T) -> Option<T> {
        if self { Some(t) } else { None }
    }

    pub fn then<T, F: FnOnce() -> T>(self, f: F) -> Option<T> {
        if self { Some(f()) } else { None }
    }
}
// endregion:bool_impl

// region:int_impl
macro_rules! impl_int {
    ($($t:ty)*) => {
        $(
            impl $t {
                pub const fn from_ne_bytes(bytes: [u8; size_of::<Self>()]) -> Self {
                    unsafe { mem::transmute(bytes) }
                }
            }
        )*
    }
}

impl_int! {
    usize u8 u16 u32 u64 u128
    isize i8 i16 i32 i64 i128
}
// endregion:int_impl

// region:error
pub mod error {
    #[rustc_has_incoherent_inherent_impls]
    pub trait Error: crate::fmt::Debug + crate::fmt::Display {
        fn source(&self) -> Option<&(dyn Error + 'static)> {
            None
        }
    }
}
// endregion:error

// region:column
#[rustc_builtin_macro]
#[macro_export]
macro_rules! column {
    () => {};
}
// endregion:column

pub mod prelude {
    pub mod v1 {
        pub use crate::{
            clone::Clone,                                 // :clone
            cmp::{Eq, PartialEq},                         // :eq
            cmp::{Ord, PartialOrd},                       // :ord
            convert::AsMut,                               // :as_mut
            convert::AsRef,                               // :as_ref
            convert::{From, Into, TryFrom, TryInto},      // :from
            default::Default,                             // :default
            fmt::derive::Debug,                           // :fmt, derive
            hash::derive::Hash,                           // :hash, derive
            iter::{FromIterator, IntoIterator, Iterator}, // :iterator
            macros::builtin::{derive, derive_const},      // :derive
            marker::Copy,                                 // :copy
            marker::Send,                                 // :send
            marker::Sized,                                // :sized
            marker::Sync,                                 // :sync
            mem::drop,                                    // :drop
            mem::size_of,                                 // :size_of
            ops::Drop,                                    // :drop
            ops::{AsyncFn, AsyncFnMut, AsyncFnOnce},      // :async_fn
            ops::{Fn, FnMut, FnOnce},                     // :fn
            option::Option::{self, None, Some},           // :option
            panic,                                        // :panic
            result::Result::{self, Err, Ok},              // :result
            str::FromStr,                                 // :str
        };
    }

    pub mod rust_2015 {
        pub use super::v1::*;
    }

    pub mod rust_2018 {
        pub use super::v1::*;
    }

    pub mod rust_2021 {
        pub use super::v1::*;
    }

    pub mod rust_2024 {
        pub use super::v1::*;
    }
}

#[prelude_import]
#[allow(unused)]
use prelude::v1::*;
