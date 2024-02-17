//@ check-pass

#![crate_type = "lib"]
#![feature(no_core, lang_items, unboxed_closures, auto_traits, intrinsics, rustc_attrs, staged_api)]
#![feature(fundamental)]
#![feature(const_trait_impl, effects, const_mut_refs)]
#![allow(internal_features)]
#![no_std]
#![no_core]
#![stable(feature = "minicore", since = "1.0.0")]

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

#[lang = "add"]
#[const_trait]
trait Add<Rhs = Self> {
    type Output;

    fn add(self, rhs: Rhs) -> Self::Output;
}

impl const Add for i32 {
    type Output = i32;
    fn add(self, rhs: i32) -> i32 {
        loop {}
    }
}

fn foo() {
    let x = 42_i32 + 43_i32;
}

const fn bar() {
    let x = 42_i32 + 43_i32;
}


#[lang = "Try"]
#[const_trait]
trait Try: FromResidual<Self::Residual> {
    type Output;
    type Residual;

    #[lang = "from_output"]
    fn from_output(output: Self::Output) -> Self;

    #[lang = "branch"]
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output>;
}

// FIXME
// #[const_trait]
trait FromResidual<R = <Self as /* FIXME: ~const */ Try>::Residual> {
    #[lang = "from_residual"]
    fn from_residual(residual: R) -> Self;
}

enum ControlFlow<B, C = ()> {
    #[lang = "Continue"]
    Continue(C),
    #[lang = "Break"]
    Break(B),
}

#[const_trait]
#[lang = "fn"]
#[rustc_paren_sugar]
trait Fn<Args: Tuple>: ~const FnMut<Args> {
    extern "rust-call" fn call(&self, args: Args) -> Self::Output;
}

#[const_trait]
#[lang = "fn_mut"]
#[rustc_paren_sugar]
trait FnMut<Args: Tuple>: ~const FnOnce<Args> {
    extern "rust-call" fn call_mut(&mut self, args: Args) -> Self::Output;
}

#[const_trait]
#[lang = "fn_once"]
#[rustc_paren_sugar]
trait FnOnce<Args: Tuple> {
    #[lang = "fn_once_output"]
    type Output;

    extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
}

struct ConstFnMutClosure<CapturedData, Function> {
    data: CapturedData,
    func: Function,
}

#[lang = "tuple_trait"]
trait Tuple {}

macro_rules! impl_fn_mut_tuple {
    ($($var:ident)*) => {
        impl<'a, $($var,)* ClosureArguments: Tuple, Function, ClosureReturnValue> const
            FnOnce<ClosureArguments> for ConstFnMutClosure<($(&'a mut $var),*), Function>
        where
            Function: ~const Fn(($(&mut $var),*), ClosureArguments) -> ClosureReturnValue,
            Function: ~const Destruct,
        {
            type Output = ClosureReturnValue;

            extern "rust-call" fn call_once(mut self, args: ClosureArguments) -> Self::Output {
            self.call_mut(args)
            }
        }
        impl<'a, $($var,)* ClosureArguments: Tuple, Function, ClosureReturnValue> const
            FnMut<ClosureArguments> for ConstFnMutClosure<($(&'a mut $var),*), Function>
        where
            Function: ~const Fn(($(&mut $var),*), ClosureArguments)-> ClosureReturnValue,
            Function: ~const Destruct,
        {
            extern "rust-call" fn call_mut(&mut self, args: ClosureArguments) -> Self::Output {
                #[allow(non_snake_case)]
                let ($($var),*) = &mut self.data;
                (self.func)(($($var),*), args)
            }
        }
    };
}
//impl_fn_mut_tuple!(A);
//impl_fn_mut_tuple!(A B);
//impl_fn_mut_tuple!(A B C);
//impl_fn_mut_tuple!(A B C D);
//impl_fn_mut_tuple!(A B C D E);

#[lang = "receiver"]
trait Receiver {}

impl<T: ?Sized> Receiver for &T {}

impl<T: ?Sized> Receiver for &mut T {}

#[lang = "destruct"]
#[const_trait]
trait Destruct {}

#[lang = "freeze"]
unsafe auto trait Freeze {}

#[lang = "drop"]
#[const_trait]
trait Drop {
    fn drop(&mut self);
}

/*
#[const_trait]
trait Residual<O> {
    type TryType: ~const Try<Output = O, Residual = Self> + Try<Output = O, Residual = Self>;
}
*/

const fn size_of<T>() -> usize {
    42
}

impl Copy for u8 {}

impl usize {
    #[rustc_allow_incoherent_impl]
    const fn repeat_u8(x: u8) -> usize {
        usize::from_ne_bytes([x; size_of::<usize>()])
    }
    #[rustc_allow_incoherent_impl]
    const fn from_ne_bytes(bytes: [u8; size_of::<Self>()]) -> Self {
        loop {}
    }
}

#[rustc_do_not_const_check] // hooked by const-eval
const fn panic_display() {
    panic_fmt();
}

fn panic_fmt() {}

#[lang = "index"]
#[const_trait]
trait Index<Idx: ?Sized> {
    type Output: ?Sized;

    fn index(&self, index: Idx) -> &Self::Output;
}


#[const_trait]
unsafe trait SliceIndex<T: ?Sized> {
    type Output: ?Sized;
    fn index(self, slice: &T) -> &Self::Output;
}

impl<T, I> const Index<I> for [T]
where
    I: ~const SliceIndex<[T]>,
{
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &I::Output {
        index.index(self)
    }
}
/* FIXME
impl<T, I, const N: usize> const Index<I> for [T; N]
where
    [T]: ~const Index<I>,
{
    type Output = <[T] as Index<I>>::Output;

    #[inline]
    // FIXME: make `Self::Output` act like `<Self as ~const Index<I>>::Output`
    fn index(&self, index: I) -> &<[T] as Index<I>>::Output {
        Index::index(self as &[T], index)
    }
}
*/

#[lang = "unsize"]
trait Unsize<T: ?Sized> {
}

#[lang = "coerce_unsized"]
trait CoerceUnsized<T: ?Sized> {
}

impl<'a, 'b: 'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b T {}


#[lang = "deref"]
// #[const_trait] FIXME
trait Deref {
    #[lang = "deref_target"]
    type Target: ?Sized;

    fn deref(&self) -> &Self::Target;
}


impl<T: ?Sized> /* const */ Deref for &T {
    type Target = T;

    fn deref(&self) -> &T {
        *self
    }
}

impl<T: ?Sized> /* const */ Deref for &mut T {
    type Target = T;

    fn deref(&self) -> &T {
        *self
    }
}

enum Option<T> {
    #[lang = "None"]
    None,
    #[lang = "Some"]
    Some(T),
}

impl<T> Option<T> {
    const fn as_ref(&self) -> Option<&T> {
        match *self {
            Some(ref x) => Some(x),
            None => None,
        }
    }

    const fn as_mut(&mut self) -> Option<&mut T> {
        match *self {
            Some(ref mut x) => Some(x),
            None => None,
        }
    }
}

use Option::*;

/*
const fn as_deref<T>(opt: &Option<T>) -> Option<&T::Target>
where
    T: ~const Deref,
{
    match opt {
        Option::Some(t) => Option::Some(t.deref()),
        Option::None => Option::None,
    }
}
*/

#[const_trait]
trait Into<T>: Sized {
    fn into(self) -> T;
}

#[const_trait]
trait From<T>: Sized {
    fn from(value: T) -> Self;
}

impl<T, U> const Into<U> for T
where
    U: ~const From<T>,
{
    fn into(self) -> U {
        U::from(self)
    }
}

impl<T> const From<T> for T {
    fn from(t: T) -> T {
        t
    }
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}
use Result::*;

fn from_str(s: &str) -> Result<bool, ()> {
    match s {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(()),
    }
}

#[lang = "eq"]
#[const_trait]
trait PartialEq<Rhs: ?Sized = Self> {
    fn eq(&self, other: &Rhs) -> bool;
    fn ne(&self, other: &Rhs) -> bool {
        !self.eq(other)
    }
}

impl<A: ?Sized, B: ?Sized> const PartialEq<&B> for &A
where
    A: ~const PartialEq<B>,
{
    fn eq(&self, other: &&B) -> bool {
        PartialEq::eq(*self, *other)
    }
}

impl PartialEq for str {
    fn eq(&self, other: &str) -> bool {
        loop {}
    }
}


#[lang = "not"]
#[const_trait]
trait Not {
    type Output;
    fn not(self) -> Self::Output;
}

impl const Not for bool {
    type Output = bool;
    fn not(self) -> bool {
        !self
    }
}

impl Copy for bool {}
impl<'a> Copy for &'a str {}

#[lang = "pin"]
#[fundamental]
#[repr(transparent)]
struct Pin<P> {
    pointer: P,
}

impl<P> Pin<P> {
    #[lang = "new_unchecked"]
    const unsafe fn new_unchecked(pointer: P) -> Pin<P> {
        Pin { pointer }
    }
}

impl<'a, T: ?Sized> Pin<&'a T> {
    const fn get_ref(self) -> &'a T {
        self.pointer
    }
}


impl<P: Deref> Pin<P> {
    /* const */ fn as_ref(&self) -> Pin<&P::Target>
    where
        P: /* ~const */ Deref,
    {
        unsafe { Pin::new_unchecked(&*self.pointer) }
    }
}


impl<'a, T: ?Sized> Pin<&'a mut T> {
    const unsafe fn get_unchecked_mut(self) -> &'a mut T {
        self.pointer
    }
}
/* FIXME lol
impl<T> Option<T> {
    const fn as_pin_ref(self: Pin<&Self>) -> Option<Pin<&T>> {
        match Pin::get_ref(self).as_ref() {
            Some(x) => unsafe { Some(Pin::new_unchecked(x)) },
            None => None,
        }
    }

    const fn as_pin_mut(self: Pin<&mut Self>) -> Option<Pin<&mut T>> {
        unsafe {
            match Pin::get_unchecked_mut(self).as_mut() {
                Some(x) => Some(Pin::new_unchecked(x)),
                None => None,
            }
        }
    }
}
*/

impl<P: /* ~const */ Deref> /* const */ Deref for Pin<P> {
    type Target = P::Target;
    fn deref(&self) -> &P::Target {
        Pin::get_ref(Pin::as_ref(self))
    }
}

impl<T> /* const */ Deref for Option<T> {
    type Target = T;
    fn deref(&self) -> &T {
        loop {}
    }
}

impl<P: Receiver> Receiver for Pin<P> {}

impl<T: Clone> Clone for RefCell<T> {
    fn clone(&self) -> RefCell<T> {
        RefCell::new(self.borrow().clone())
    }
}

struct RefCell<T: ?Sized> {
    borrow: UnsafeCell<()>,
    value: UnsafeCell<T>,
}
impl<T> RefCell<T> {
    const fn new(value: T) -> RefCell<T> {
        loop {}
    }
}
impl<T: ?Sized> RefCell<T> {
    fn borrow(&self) -> Ref<'_, T> {
        loop {}
    }
}

#[lang = "unsafe_cell"]
#[repr(transparent)]
struct UnsafeCell<T: ?Sized> {
    value: T,
}

struct Ref<'b, T: ?Sized + 'b> {
    value: *const T,
    borrow: &'b UnsafeCell<()>,
}

impl<T: ?Sized> Deref for Ref<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        loop {}
    }
}

#[lang = "clone"]
#[rustc_trivial_field_reads]
#[const_trait]
trait Clone: Sized {
    fn clone(&self) -> Self;
    fn clone_from(&mut self, source: &Self)
    where
        Self: ~const Destruct,
    {
        *self = source.clone()
    }
}

#[lang = "structural_peq"]
trait StructuralPartialEq {}

const fn drop<T: ~const Destruct>(_: T) {}

extern "rust-intrinsic" {
    #[rustc_const_stable(feature = "const_eval_select", since = "1.0.0")]
    fn const_eval_select<ARG: Tuple, F, G, RET>(
        arg: ARG,
        called_in_const: F,
        called_at_rt: G,
    ) -> RET
    where
        F: const FnOnce<ARG, Output = RET>,
        G: FnOnce<ARG, Output = RET>;
}

fn test_const_eval_select() {
    const fn const_fn() {}
    fn rt_fn() {}

    unsafe { const_eval_select((), const_fn, rt_fn); }
}
