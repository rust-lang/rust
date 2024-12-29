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
    slf: &TemplateStruct<T>
    // Alias<Projection>   ...not Inherent. dangit
) -> TemplateStruct<T>::Out {
    slf.one + slf.two
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
       self: Box<Self>
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

extern "C" {
fn all_ty_kinds_in_ptr(
  // Ptr[UInt], Ptr[Int], Ptr[Float], Ptr[Bool]
  u: *const u8, i: *const i8, f: *const f64, b: *const bool,
  // Ptr[Struct]
  s: *const String,  //~ ERROR: uses type `String`
  // Ptr[Str]
  s2: *const str, //~ ERROR: uses type `*const str`
  // Ptr[Char]
  c: *const  char, //~ ERROR: uses type `char`
  // Ptr[Slice]
  s3: *const [u8], //~ ERROR: uses type `*const [u8]`
  // deactivated here, because this is a function *declaration* (param N unacceptable)
  // s4: *const [u8;N],
  // Ptr[Tuple]
  p: *const (u8,u8), //~ ERROR: uses type `(u8, u8)`
  // deactivated here, because this is a function *declaration* (pattern unacceptable)
  // (p2, p3):(*const u8, *const u8),
  // Pat
  nz: *const pattern_type!(u32 is 1..), //~ ERROR: uses type `(u32) is 1..=`
  // deactivated here, because this is a function *declaration* (pattern unacceptable)
  //SomeStruct{b: ref p4,..}: & SomeStruct,
  // Ptr[Union]
  u2: *const SomeUnion,
  // Ptr[Enum],
  e: *const SomeEnum,
  // deactivated here, because this is a function *declaration* (impl type unacceptable)
  //d: *const impl Clone,
  // deactivated here, because this is a function *declaration* (type param unacceptable)
  //t: *const T,
  // Ptr[Foreign]
  e2: *mut ExtType,
  // Ptr[Struct]
  e3: *const StructWithDyn, //~ ERROR: uses type `*const StructWithDyn`
  // Ptr[Never]
  x: *const !,
  //r1: &u8, r2: *const u8, r3: Box<u8>,
  // Ptr[FnPtr]
  f2: *const fn(u8)->u8, //~ ERROR: uses type `fn(u8) -> u8`
  // Ptr[Dynamic]
  f3: *const dyn Fn(u8)->u8, //~ ERROR: uses type `*const dyn Fn(u8) -> u8`
  // Ptr[Dynamic]
  d2: *const dyn std::cmp::PartialOrd<u8>, //~ ERROR: uses type `*const dyn PartialOrd<u8>`
  // deactivated here, because this is a function *declaration*  (impl type unacceptable)
  //a: *const impl async Fn(u8)->u8,
  // Alias<Opaque> (this gets caught outside of the code we want to test)
) -> *const dyn std::fmt::Debug; //~ ERROR: uses type `*const dyn Debug`
}

extern "C" fn all_ty_kinds_in_ref<'a, const N:usize,T>(
  // Ref[UInt], Ref[Int], Ref[Float], Ref[Bool]
  u: &u8, i: &'a i8, f: &f64, b: &bool,
  // Ref[Struct]
  s: &String,
  // Ref[Str]
  s2: &str, //~ ERROR: uses type `&str`
  // Ref[Char]
  c: &char,
  // Ref[Slice]
  s3: &[u8], //~ ERROR: uses type `&[u8]`
  // Ref[Array] (this gets caught outside of the code we want to test)
  s4: &[u8;N],
  // Ref[Tuple]
  p: &(u8, u8),
  // also Tuple
  (p2, p3):(&u8, &u8), //~ ERROR: uses type `(&u8, &u8)`
  // Pat
  nz: &pattern_type!(u32 is 1..),
  // Ref[Struct]
  SomeStruct{b: ref p4,..}: &SomeStruct,
  // Ref[Union]
  u2:  &SomeUnion,
  // Ref[Enum],
  e:  &SomeEnum,
  // Ref[Param]
  d:  &impl Clone,
  // Ref[Param]
  t:  &T,
  // Ref[Foreign]
  e2: &ExtType,
  // Ref[Struct]
  e3: &StructWithDyn, //~ ERROR: uses type `&StructWithDyn`
  // Ref[Never]
  x: &!,
  //r1: &u8, r2:  &u8, r3: Box<u8>,
  // Ref[FnPtr]
  f2: &fn(u8)->u8,
  // Ref[Dynamic]
  f3: &dyn Fn(u8)->u8, //~ ERROR: uses type `&dyn Fn(u8) -> u8`
  // Ref[Dynamic]
  d2: &dyn std::cmp::PartialOrd<u8>, //~ ERROR: uses type `&dyn PartialOrd<u8>`
  // Ref[Param],
  a: &impl async Fn(u8)->u8,
  // Ref[Dynamic] (this gets caught outside of the code we want to test)
) -> &'a dyn std::fmt::Debug { //~ ERROR: uses type `&dyn Debug`
    i
}

extern "C" fn all_ty_kinds_in_box<const N:usize,T>(
  // Box[UInt], Box[Int], Box[Float], Box[Bool]
  u: Box<u8>, i: Box<i8>, f: Box<f64>, b: Box<bool>,
  // Box[Struct]
  s: Box<String>,
  // Box[Str]
  s2: Box<str>, //~ ERROR: uses type `Box<str>`
  // Box[Char]
  c: Box<char>,
  // Box[Slice]
  s3: Box<[u8]>, //~ ERROR: uses type `Box<[u8]>`
  // Box[Array] (this gets caught outside of the code we want to test)
  s4: Box<[u8;N]>,
  // Box[Tuple]
  p: Box<(u8,u8)>,
  // also Tuple
  (p2,p3):(Box<u8>, Box<u8>), //~ ERROR: uses type `(Box<u8>, Box<u8>)`
  // Pat
  nz: Box<pattern_type!(u32 is 1..)>,
  // Ref[Struct]
  SomeStruct{b: ref p4,..}: &SomeStruct,
  // Box[Union]
  u2:  Box<SomeUnion>,
  // Box[Enum],
  e:  Box<SomeEnum>,
  // Box[Param]
  d:  Box<impl Clone>,
  // Box[Param]
  t:  Box<T>,
  // Box[Foreign]
  e2: Box<ExtType>,
  // Box[Struct]
  e3: Box<StructWithDyn>, //~ ERROR: uses type `Box<StructWithDyn>`
  // Box[Never]
  x: Box<!>,
  //r1: Box<u8, r2:  Box<u8, r3: Box<u8>,
  // Box[FnPtr]
  f2: Box<fn(u8)->u8>,
  // Box[Dynamic]
  f3: Box<dyn Fn(u8)->u8>,  //~ ERROR: uses type `Box<dyn Fn(u8) -> u8>`
  // Box[Dynamic]
  d2: Box<dyn std::cmp::PartialOrd<u8>>, //~ ERROR: uses type `Box<dyn PartialOrd<u8>>`
  // Box[Param],
  a: Box<impl async Fn(u8)->u8>,
  // Box[Dynamic] (this gets caught outside of the code we want to test)
) -> Box<dyn std::fmt::Debug> { //~ ERROR: uses type `Box<dyn Debug>`
    i
}

fn main() {}
