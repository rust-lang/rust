// Trying to cover as many ty_kinds as possible in the code for ImproperCTypes lint
//@ edition:2018

#![allow(dead_code,unused_variables)]
#![deny(improper_ctypes,improper_ctypes_definitions)]

// we want ALL the ty_kinds, including the feature-gated ones
#![feature(extern_types)]
#![feature(never_type)]
#![feature(inherent_associated_types)] //~ WARN: is incomplete
#![feature(async_trait_bounds)]
#![feature(pattern_types, rustc_attrs)]
#![feature(pattern_type_macro)]

// ty_kinds not found so far:
// Placeholder, Bound, Infer, Error,
// Alias<Weak|Inherent>
// FnDef, Closure, Coroutine, ClosureCoroutine, CoroutineWitness,

use std::ptr::from_ref;
use std::ptr::NonNull;
use std::mem::{MaybeUninit, size_of};
use std::num::NonZero;
use std::pat::pattern_type;

#[repr(C)]
struct SomeStruct{
  a: u8,
  b: i32,
}
impl SomeStruct{
  extern "C" fn klol(
      // Ref[Struct]
      &self
  ){}
}

#[repr(C)]
#[derive(Clone,Copy)]
struct TemplateStruct<T> where T: std::ops::Add+Copy {
   one: T,
   two: T,
}
impl<T: std::ops::Add+Copy> TemplateStruct<T> {
    type Out = <T as std::ops::Add>::Output;
}

extern "C" fn tstruct_sum<T: std::ops::Add+Copy>(
    // Ref[Struct]
    slf: Option<&TemplateStruct<T>>
    // Option<Alias<Projection>>   ...not Inherent. dangit
) -> Option<Box<TemplateStruct<T>::Out>> {
    Some(Box::new(slf?.one + slf?.two))
}

#[repr(C)]
union SomeUnion{
   sz: u8,
   us: i8,
}
#[repr(C)]
enum SomeEnum{
   Everything=42,
   NotAU8=256,
   SomePrimeNumber=23,
}

pub trait TimesTwo: std::ops::Add<Self> + Sized + Clone
   where for<'a> &'a Self: std::ops::Add<&'a Self>,
         *const Self: std::ops::Add<*const Self>,
         Box<Self>: std::ops::Add<Box<Self>>,
{
   extern "C" fn t2_own(
       // Param
       self
       // Alias<Projection>
   ) -> <Self as std::ops::Add<Self>>::Output {
       self.clone() + self
   }
   // it ICEs (https://github.com/rust-lang/rust/issues/134587) :(
   //extern "C" fn t2_ptr(
   //    // Ref[Param]
   //    slf: *const Self
   //    // Alias<Projection>
   //) -> <*const Self as std::ops::Add<*const Self>>::Output {
   //    slf + slf
   //}
   extern "C" fn t2_box(
       // Box[Param]
       self: Box<Self>,
       // Alias<Projection>
   ) -> <Box<Self> as std::ops::Add<Box<Self>>>::Output {
       self.clone() + self
   }
   extern "C" fn t2_ref(
       // Ref[Param]
       &self
       // Alias<Projection>
       ) -> <&Self as std::ops::Add<&Self>>::Output {
       self + self
   }
}

extern "C" {type ExtType;}

#[repr(C)]
pub struct StructWithDyn(dyn std::fmt::Debug);

extern "C" {
  // variadic args aren't listed as args in a way that allows type checking.
  // this is fine (TM)
  fn variadic_function(e: ...);
}

extern "C" fn all_ty_kinds<'a,const N:usize,T>(
  // UInt, Int, Float, Bool
  u:u8, i:i8, f:f64, b:bool,
  // Struct
  s:String, //~ ERROR: uses type `String`
  // Ref[Str]
  s2:&str, //~ ERROR: uses type `&str`
  // Char
  c: char,  //~ ERROR: uses type `char`
  // Ref[Slice]
  s3:&[u8], //~ ERROR: uses type `&[u8]`
  // Array (this gets caught outside of the code we want to test)
  s4:[u8;N], //~ ERROR: uses type `[u8; N]`
  // Tuple
  p:(u8, u8), //~ ERROR: uses type `(u8, u8)`
  // also Tuple
  (p2, p3):(u8, u8), //~ ERROR: uses type `(u8, u8)`
  // Pat
  nz: pattern_type!(u32 is 1..), //~ ERROR: uses type `(u32) is 1..=`
  // Struct
  SomeStruct{b:p4,..}: SomeStruct,
  // Union
  u2: SomeUnion,
  // Enum,
  e: SomeEnum,
  // Param
  d: impl Clone,
  // Param
  t: T,
  // Ptr[Foreign]
  e2: *mut ExtType,
  // Ref[Struct]
  e3: &StructWithDyn, //~ ERROR: uses type `&StructWithDyn`
  // Never
  x:!,
  //r1: &u8, r2: *const u8, r3: Box<u8>,
  // FnPtr
  f2: fn(u8)->u8, //~ ERROR: uses type `fn(u8) -> u8`
  // Ref[Dynamic]
  f3: &'a dyn Fn(u8)->u8, //~ ERROR: uses type `&dyn Fn(u8) -> u8`
  // Ref[Dynamic]
  d2: &dyn std::cmp::PartialOrd<u8>, //~ ERROR: uses type `&dyn PartialOrd<u8>`
  // Param,
  a: impl async Fn(u8)->u8,  //FIXME: eventually, be able to peer into type params
  // Alias<Opaque> (this gets caught outside of the code we want to test)
) -> impl std::fmt::Debug { //~ ERROR: uses type `impl Debug`
    3_usize
}

extern "C" fn all_ty_kinds_in_ptr<const N:usize, T>(
  // Ptr[UInt], Ptr[Int], Ptr[Float], Ptr[Bool]
  u: *const u8, i: *const i8, f: *const f64, b: *const bool,
  // Ptr[Struct]
  s: *const String,
  // Ptr[Str]
  s2: *const str, //~ ERROR: uses type `*const str`
  // Ptr[Char]
  c: *const char,
  // Ptr[Slice]
  s3: *const [u8], //~ ERROR: uses type `*const [u8]`
  // Ptr[Array] (this gets caught outside of the code we want to test)
  s4: *const [u8;N],
  // Ptr[Tuple]
  p: *const (u8,u8),
  // Tuple
  (p2, p3):(*const u8, *const u8),
  // Pat
  nz: *const pattern_type!(u32 is 1..), //~ ERROR: uses type `(u32) is 1..=`
  // Ptr[Struct]
  SomeStruct{b: ref p4,..}: & SomeStruct,
  // Ptr[Union]
  u2: *const SomeUnion,
  // Ptr[Enum],
  e: *const SomeEnum,
  // Param
  d: *const impl Clone,
  // Param
  t: *const T,
  // Ptr[Foreign]
  e2: *mut ExtType,
  // Ptr[Struct]
  e3: *const StructWithDyn, //~ ERROR: uses type `*const StructWithDyn`
  // Ptr[Never]
  x: *const !,
  //r1: &u8, r2: *const u8, r3: Box<u8>,
  // Ptr[FnPtr]
  f2: *const fn(u8)->u8,
  // Ptr[Dynamic]
  f3: *const dyn Fn(u8)->u8, //~ ERROR: uses type `*const dyn Fn(u8) -> u8`
  // Ptr[Dynamic]
  d2: *const dyn std::cmp::PartialOrd<u8>, //~ ERROR: uses type `*const dyn PartialOrd<u8>`
  // Ptr[Param],
  a: *const impl async Fn(u8)->u8,
  // Alias<Opaque> (this gets caught outside of the code we want to test)
) -> *const dyn std::fmt::Debug { //~ ERROR: uses type `*const dyn Debug`
    todo!()
}

extern "C" {
fn all_ty_kinds_in_ref<'a>(
  // Ref[UInt], Ref[Int], Ref[Float], Ref[Bool]
  u: &u8, i: &'a i8, f: &f64, b: &bool,
  // Ref[Struct]
  s: &String, //~ ERROR: uses type `String`
  // Ref[Str]
  s2: &str, //~ ERROR: uses type `&str`
  // Ref[Char]
  c: &char, //~ ERROR: uses type `char`
  // Ref[Slice]
  s3: &[u8], //~ ERROR: uses type `&[u8]`
  // deactivated here, because this is a function *declaration* (param N unacceptable)
  // s4: &[u8;N],
  // Ref[Tuple]
  p: &(u8, u8), //~ ERROR: uses type `(u8, u8)`
  // deactivated here, because this is a function *declaration* (patterns unacceptable)
  // (p2, p3):(&u8, &u8), //~ ERROR: uses type `(&u8, &u8)`
  // Pat
  nz: &pattern_type!(u32 is 1..),
  // deactivated here, because this is a function *declaration* (pattern unacceptable)
  // SomeStruct{b: ref p4,..}: &SomeStruct,
  // Ref[Union]
  u2:  &SomeUnion,
  // Ref[Enum],
  e:  &SomeEnum,
  // deactivated here, because this is a function *declaration* (impl type unacceptable)
  // d: &impl Clone,
  // deactivated here, because this is a function *declaration* (type param unacceptable)
  // t: &T,
  // Ref[Foreign]
  e2: &ExtType,
  // Ref[Struct]
  e3: &StructWithDyn, //~ ERROR: uses type `&StructWithDyn`
  // Ref[Never]
  x: &!,
  //r1: &u8, r2:  &u8, r3: Box<u8>,
  // Ref[FnPtr]
  f2: &fn(u8)->u8, //~ ERROR: uses type `fn(u8) -> u8`
  // Ref[Dynamic]
  f3: &dyn Fn(u8)->u8, //~ ERROR: uses type `&dyn Fn(u8) -> u8`
  // Ref[Dynamic]
  d2: &dyn std::cmp::PartialOrd<u8>, //~ ERROR: uses type `&dyn PartialOrd<u8>`
  // deactivated here, because this is a function *declaration*  (impl type unacceptable)
  // a: &impl async Fn(u8)->u8,
  // Ref[Dynamic] (this gets caught outside of the code we want to test)
) -> &'a dyn std::fmt::Debug; //~ ERROR: uses type `&dyn Debug`
}

extern "C" fn all_ty_kinds_in_box<const N:usize,T>(
  // Box[UInt], Box[Int], Box[Float], Box[Bool]
  u: Option<Box<u8>>, i: Option<Box<i8>>, f: Option<Box<f64>>, b: Option<Box<bool>>,
  // Box[Struct]
  s: Option<Box<String>>,
  // Box[Str]
  s2: Box<str>, //~ ERROR: uses type `Box<str>`
  // Box[Char]
  c: Box<char>, //~ ERROR: uses type `Box<char>`
  // Box[Slice]
  s3: Box<[u8]>, //~ ERROR: uses type `Box<[u8]>`
  // Box[Array] (this gets caught outside of the code we want to test)
  s4: Option<Box<[u8;N]>>,
  // Box[Tuple]
  p: Option<Box<(u8,u8)>>,
  // also Tuple
  (p2,p3):(Box<u8>, Box<u8>), //~ ERROR: uses type `(Box<u8>, Box<u8>)`
  // Pat
  nz: Option<Box<pattern_type!(u32 is 1..)>>,
  // Ref[Struct]
  SomeStruct{b: ref p4,..}: &SomeStruct,
  // Box[Union]
  u2: Option<Box<SomeUnion>>,
  // Box[Enum],
  e: Option<Box<SomeEnum>>,
  // Box[Param]
  d: Option<Box<impl Clone>>,
  // Box[Param]
  t: Option<Box<T>>,
  // Box[Foreign]
  e2: Option<Box<ExtType>>,
  // Box[Struct]
  e3: Box<StructWithDyn>, //~ ERROR: uses type `Box<StructWithDyn>`
  // Box[Never]
  // (considered FFI-unsafe because of null pointers, not the litteral uninhabited type. smh.)
  x: Box<!>, //~ ERROR: uses type `Box<!>`
  //r1: Box<u8, r2:  Box<u8, r3: Box<u8>,
  // Box[FnPtr]
  f2: Box<fn(u8)->u8>,  //~ ERROR: uses type `Box<fn(u8) -> u8>`
  // Box[Dynamic]
  f3: Box<dyn Fn(u8)->u8>,  //~ ERROR: uses type `Box<dyn Fn(u8) -> u8>`
  // Box[Dynamic]
  d2: Box<dyn std::cmp::PartialOrd<u8>>, //~ ERROR: uses type `Box<dyn PartialOrd<u8>>`
  // Option[Box[Param]],
  a: Option<Box<impl async Fn(u8)->u8>>,
  // Box[Dynamic] (this gets caught outside of the code we want to test)
) -> Box<dyn std::fmt::Debug> { //~ ERROR: uses type `Box<dyn Debug>`
    u.unwrap()
}

fn main() {}
