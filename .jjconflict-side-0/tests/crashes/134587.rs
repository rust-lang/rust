//@ known-bug: #134587

use std::ops::Add;

pub fn foo<T>(slf: *const T)
where
    *const T: Add,
{
    slf + slf;
}

pub fn foo2<T>(slf: *const T)
where
    *const T: Add<u8>,
{
    slf + 1_u8;
}


pub trait TimesTwo
   where *const Self: Add<*const Self>,
{
   extern "C" fn t2_ptr(slf: *const Self)
   -> <*const Self as Add<*const Self>>::Output {
       slf + slf
   }
}
