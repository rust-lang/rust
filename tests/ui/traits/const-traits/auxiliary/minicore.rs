//@ compile-flags: -Znext-solver -Cpanic=abort
//@ no-prefer-dynamic

#![crate_type = "rlib"]
#![feature(
    no_core,
    lang_items,
    unboxed_closures,
    auto_traits,
    intrinsics,
    rustc_attrs,
    fundamental,
    marker_trait_attr,
    const_trait_impl,
    const_destruct,
)]
#![allow(internal_features, incomplete_features)]
#![no_std]
#![no_core]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

#[lang = "copy"]
pub trait Copy {}

impl Copy for bool {}
impl Copy for u8 {}
impl<T: PointeeSized> Copy for &T {}

#[lang = "add"]
#[const_trait]
pub trait Add<Rhs = Self> {
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
pub trait Try: FromResidual<Self::Residual> {
    type Output;
    type Residual;

    #[lang = "from_output"]
    fn from_output(output: Self::Output) -> Self;

    #[lang = "branch"]
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output>;
}

#[const_trait]
pub trait FromResidual<R = <Self as Try>::Residual> {
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
pub trait Fn<Args: Tuple>: [const] FnMut<Args> {
    extern "rust-call" fn call(&self, args: Args) -> Self::Output;
}

#[const_trait]
#[lang = "fn_mut"]
#[rustc_paren_sugar]
pub trait FnMut<Args: Tuple>: [const] FnOnce<Args> {
    extern "rust-call" fn call_mut(&mut self, args: Args) -> Self::Output;
}

#[const_trait]
#[lang = "fn_once"]
#[rustc_paren_sugar]
pub trait FnOnce<Args: Tuple> {
    #[lang = "fn_once_output"]
    type Output;

    extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
}

#[lang = "tuple_trait"]
pub trait Tuple {}

#[lang = "legacy_receiver"]
pub trait LegacyReceiver {}

impl<T: PointeeSized> LegacyReceiver for &T {}

impl<T: PointeeSized> LegacyReceiver for &mut T {}

#[lang = "receiver"]
pub trait Receiver {
    #[lang = "receiver_target"]
    type Target: MetaSized;
}

impl<T: Deref + MetaSized> Receiver for T {
    type Target = <T as Deref>::Target;
}

#[lang = "destruct"]
#[const_trait]
pub trait Destruct {}

#[lang = "freeze"]
pub unsafe auto trait Freeze {}

#[lang = "drop"]
#[const_trait]
pub trait Drop {
    fn drop(&mut self);
}

#[const_trait]
pub trait Residual<O> {
    type TryType: [const] Try<Output = O, Residual = Self> + Try<Output = O, Residual = Self>;
}

const fn size_of<T>() -> usize {
    42
}

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
pub trait Index<Idx: PointeeSized> {
    type Output: MetaSized;

    fn index(&self, index: Idx) -> &Self::Output;
}

#[const_trait]
pub unsafe trait SliceIndex<T: PointeeSized> {
    type Output: MetaSized;
    fn index(self, slice: &T) -> &Self::Output;
}

impl<T, I> const Index<I> for [T]
where
    I: [const] SliceIndex<[T]>,
{
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &I::Output {
        index.index(self)
    }
}

impl<T, I, const N: usize> const Index<I> for [T; N]
where
    [T]: [const] Index<I>,
{
    type Output = <[T] as Index<I>>::Output;

    #[inline]
    fn index(&self, index: I) -> &<[T] as Index<I>>::Output {
        Index::index(self as &[T], index)
    }
}

#[lang = "unsize"]
pub trait Unsize<T: PointeeSized>: PointeeSized {}

#[lang = "coerce_unsized"]
pub trait CoerceUnsized<T: PointeeSized> {}

impl<'a, 'b: 'a, T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<&'a U> for &'b T {}

#[lang = "deref"]
#[const_trait]
pub trait Deref {
    #[lang = "deref_target"]
    type Target: MetaSized;

    fn deref(&self) -> &Self::Target;
}

impl<T: MetaSized> const Deref for &T {
    type Target = T;

    fn deref(&self) -> &T {
        *self
    }
}

impl<T: MetaSized> const Deref for &mut T {
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

const fn as_deref<T>(opt: &Option<T>) -> Option<&T::Target>
where
    T: [const] Deref,
{
    match opt {
        Option::Some(t) => Option::Some(t.deref()),
        Option::None => Option::None,
    }
}

#[const_trait]
pub trait Into<T>: Sized {
    fn into(self) -> T;
}

#[const_trait]
pub trait From<T>: Sized {
    fn from(value: T) -> Self;
}

impl<T, U> const Into<U> for T
where
    U: [const] From<T>,
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
pub trait PartialEq<Rhs: PointeeSized = Self>: PointeeSized {
    fn eq(&self, other: &Rhs) -> bool;
    fn ne(&self, other: &Rhs) -> bool {
        !self.eq(other)
    }
}

impl<A: PointeeSized, B: PointeeSized> const PartialEq<&B> for &A
where
    A: [const] PartialEq<B>,
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
pub trait Not {
    type Output;
    fn not(self) -> Self::Output;
}

impl const Not for bool {
    type Output = bool;
    fn not(self) -> bool {
        !self
    }
}

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

impl<'a, T: PointeeSized> Pin<&'a T> {
    const fn get_ref(self) -> &'a T {
        self.pointer
    }
}

impl<P: Deref> Pin<P> {
    const fn as_ref(&self) -> Pin<&P::Target>
    where
        P: [const] Deref,
    {
        unsafe { Pin::new_unchecked(&*self.pointer) }
    }
}

impl<'a, T: PointeeSized> Pin<&'a mut T> {
    const unsafe fn get_unchecked_mut(self) -> &'a mut T {
        self.pointer
    }
}

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

impl<P: [const] Deref> const Deref for Pin<P> {
    type Target = P::Target;
    fn deref(&self) -> &P::Target {
        Pin::get_ref(Pin::as_ref(self))
    }
}

impl<T> const Deref for Option<T> {
    type Target = T;
    fn deref(&self) -> &T {
        loop {}
    }
}

impl<P: LegacyReceiver> LegacyReceiver for Pin<P> {}

impl<T: Clone> Clone for RefCell<T> {
    fn clone(&self) -> RefCell<T> {
        RefCell::new(self.borrow().clone())
    }
}

struct RefCell<T: PointeeSized> {
    borrow: UnsafeCell<()>,
    value: UnsafeCell<T>,
}
impl<T> RefCell<T> {
    const fn new(value: T) -> RefCell<T> {
        loop {}
    }
}
impl<T: PointeeSized> RefCell<T> {
    fn borrow(&self) -> Ref<'_, T> {
        loop {}
    }
}

#[lang = "unsafe_cell"]
#[repr(transparent)]
struct UnsafeCell<T: PointeeSized> {
    value: T,
}

struct Ref<'b, T: PointeeSized + 'b> {
    value: *const T,
    borrow: &'b UnsafeCell<()>,
}

impl<T: MetaSized> Deref for Ref<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        loop {}
    }
}

#[lang = "clone"]
#[rustc_trivial_field_reads]
#[const_trait]
pub trait Clone: Sized {
    fn clone(&self) -> Self;
    fn clone_from(&mut self, source: &Self)
    where
        Self: [const] Destruct,
    {
        *self = source.clone()
    }
}

#[lang = "structural_peq"]
pub trait StructuralPartialEq {}

pub const fn drop<T: [const] Destruct>(_: T) {}

#[rustc_intrinsic]
const fn const_eval_select<ARG: Tuple, F, G, RET>(
    arg: ARG,
    called_in_const: F,
    called_at_rt: G,
) -> RET
where
    F: const FnOnce<ARG, Output = RET>,
    G: FnOnce<ARG, Output = RET>;
