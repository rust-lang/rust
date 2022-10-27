// check-pass
use std::cell::UnsafeCell;

// R->R*
fn _0<T>(r: &T) -> *const T { r }
fn _1<T>(r: &T) -> *const T { r as *const _ }
fn _2<T>(r: &T)             { _const(r) }

// RW->RW*
fn _3<T>(r: &mut T) -> *mut T { r }
fn _4<T>(r: &mut T) -> *mut T { r as *mut _ }
fn _5<T>(r: &mut T)           { _mut(r) }

// RW->[R]->R* /!\/!\/!\
fn _6<T>(r: &mut T) -> *const T { r }             //~ warning: implicit reborrow results in a read-only pointer
fn _7<T>(r: &mut T) -> *const T { r as *const _ } //~ warning: implicit reborrow results in a read-only pointer
fn _8<T>(r: &mut T)             { _const(r) }     //~ warning: implicit reborrow results in a read-only pointer

// RW->R->R*
fn _9 <T>(r: &mut T) -> *const T { &*r }
fn _10<T>(r: &mut T) -> *const T { r as &_ as *const _ }
fn _11<T>(r: &mut T)             { _const(&r) }

// RW->RW*->R*
fn _12<T>(r: &mut T) -> *const T { r as *mut _ }
fn _13<T>(r: &mut T) -> *const T { r as *mut _ }
fn _14<T>(r: &mut T)             { _const(r as *mut _) }

// RW->[R]->R*, but the pointee is !Freeze, so be quiet,,,
fn _15(r: &mut _Water) -> *const _Water { r }
fn _16(r: &mut _Water) -> *const _Water { r as *const _ }
fn _17(r: &mut _Water)                  { _const(r) }

fn _mut(_: *mut impl Sized) {}
fn _const(_: *const impl Sized) {}

struct _Water(UnsafeCell<u8>, bool);

fn main() {}
