//@ pp-exact
//@ edition:2021

#![allow(unused_imports)]

use ::std::fmt::{self, Debug, Display, Write as _};

use core::option::Option::*;

use core::{
    cmp::{Eq, Ord, PartialEq, PartialOrd},
    convert::{AsMut, AsRef, From, Into},
    iter::{
        DoubleEndedIterator, ExactSizeIterator, Extend, FromIterator,
        IntoIterator, Iterator,
    },
    marker::{
        Copy as Copy, Send as Send, Sized as Sized, Sync as Sync, Unpin as U,
    },
    ops::{*, Drop, Fn, FnMut, FnOnce},
};

fn main() {}
