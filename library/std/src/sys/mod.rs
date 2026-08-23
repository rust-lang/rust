#![allow(unsafe_op_in_unsafe_fn)]

mod configure_builtins;
mod helpers;
mod pal;
mod personality;

pub(crate) mod alloc;
pub(crate) mod args;
pub(crate) mod backtrace;
pub(crate) mod cmath;
pub(crate) mod env;
pub(crate) mod env_consts;
pub(crate) mod exit;
pub(crate) mod fd;
pub(crate) mod fs;
pub(crate) mod io;
pub(crate) mod net;
pub(crate) mod os_str;
pub(crate) mod path;
pub(crate) mod paths;
pub(crate) mod pipe;
pub(crate) mod platform_version;
pub(crate) mod process;
pub(crate) mod random;
pub(crate) mod stdio;
pub(crate) mod sync;
pub(crate) mod thread;
pub(crate) mod thread_local;
pub(crate) mod time;

// FIXME(117276): remove this, move feature implementations into individual
//                submodules.
pub(crate) use pal::*;

/// A trait for viewing representations from std types.
#[cfg_attr(not(target_os = "linux"), allow(unused))]
pub(crate) trait AsInner<Inner: ?Sized> {
    fn as_inner(&self) -> &Inner;
}

/// A trait for viewing representations from std types.
#[cfg_attr(not(target_os = "linux"), allow(unused))]
pub(crate) trait AsInnerMut<Inner: ?Sized> {
    fn as_inner_mut(&mut self) -> &mut Inner;
}

/// A trait for extracting representations from std types.
pub(crate) trait IntoInner<Inner> {
    fn into_inner(self) -> Inner;
}

/// A trait for creating std types from internal representations.
pub(crate) trait FromInner<Inner> {
    fn from_inner(inner: Inner) -> Self;
}
