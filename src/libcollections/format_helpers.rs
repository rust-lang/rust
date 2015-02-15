// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::fmt::{self, Formatter};
use core::iter::{Iterator, IteratorExt};
use core::result::Result;

pub fn seq_fmt_debug<I: Iterator>(s: I, f: &mut Formatter) -> fmt::Result
    where I::Item:fmt::Debug
{
    for (i, e) in s.enumerate() {
        if i != 0 { try!(write!(f, ", ")); }
        try!(fmt::Debug::fmt(&e, f));
    }
    Result::Ok(())
}

pub fn seq_fmt_display<I: Iterator>(s: I, f: &mut Formatter) -> fmt::Result
    where I::Item:fmt::Display
{
    for (i, e) in s.enumerate() {
        if i != 0 { try!(write!(f, ", ")); }
        try!(fmt::Display::fmt(&e, f));
    }
    Result::Ok(())
}

pub fn seq_fmt_octal<I: Iterator>(s: I, f: &mut Formatter) -> fmt::Result
    where I::Item:fmt::Octal
{
    for (i, e) in s.enumerate() {
        if i != 0 { try!(write!(f, ", ")); }
        try!(fmt::Octal::fmt(&e, f));
    }
    Result::Ok(())
}

pub fn seq_fmt_binary<I: Iterator>(s: I, f: &mut Formatter) -> fmt::Result
    where I::Item:fmt::Binary
{
    for (i, e) in s.enumerate() {
        if i != 0 { try!(write!(f, ", ")); }
        try!(fmt::Binary::fmt(&e, f));
    }
    Result::Ok(())
}

pub fn seq_fmt_upper_hex<I: Iterator>(s: I, f: &mut Formatter) -> fmt::Result
    where I::Item:fmt::UpperHex
{
    for (i, e) in s.enumerate() {
        if i != 0 { try!(write!(f, ", ")); }
        try!(fmt::UpperHex::fmt(&e, f));
    }
    Result::Ok(())
}

pub fn seq_fmt_lower_hex<I: Iterator>(s: I, f: &mut Formatter) -> fmt::Result
    where I::Item:fmt::LowerHex
{
    for (i, e) in s.enumerate() {
        if i != 0 { try!(write!(f, ", ")); }
        try!(fmt::LowerHex::fmt(&e, f));
    }
    Result::Ok(())
}

pub fn seq_fmt_upper_exp<I: Iterator>(s: I, f: &mut Formatter) -> fmt::Result
    where I::Item:fmt::UpperExp
{
    for (i, e) in s.enumerate() {
        if i != 0 { try!(write!(f, ", ")); }
        try!(fmt::UpperExp::fmt(&e, f));
    }
    Result::Ok(())
}

pub fn seq_fmt_lower_exp<I: Iterator>(s: I, f: &mut Formatter) -> fmt::Result
    where I::Item:fmt::LowerExp
{
    for (i, e) in s.enumerate() {
        if i != 0 { try!(write!(f, ", ")); }
        try!(fmt::LowerExp::fmt(&e, f));
    }
    Result::Ok(())
}

pub fn map_fmt_debug<K, V, I: Iterator<Item=(K, V)>>(s: I, f: &mut Formatter) -> fmt::Result
    where K:fmt::Debug,
          V:fmt::Debug
{
    for (i, (k, v)) in s.enumerate() {
        if i != 0 { try!(write!(f, ", ")); }
        try!(fmt::Debug::fmt(&k, f));
        try!(write!(f, ": "));
        try!(fmt::Debug::fmt(&v, f));
    }
    Result::Ok(())
}

pub fn map_fmt_display<K, V, I: Iterator<Item=(K, V)>>(s: I, f: &mut Formatter) -> fmt::Result
    where K:fmt::Display,
          V:fmt::Display
{
    for (i, (k, v)) in s.enumerate() {
        if i != 0 { try!(write!(f, ", ")); }
        try!(fmt::Display::fmt(&k, f));
        try!(write!(f, ": "));
        try!(fmt::Display::fmt(&v, f));
    }
    Result::Ok(())
}
pub fn map_fmt_octal<K, V, I: Iterator<Item=(K, V)>>(s: I, f: &mut Formatter) -> fmt::Result
    where K:fmt::Octal,
          V:fmt::Octal
{
    for (i, (k, v)) in s.enumerate() {
        if i != 0 { try!(write!(f, ", ")); }
        try!(fmt::Octal::fmt(&k, f));
        try!(write!(f, ": "));
        try!(fmt::Octal::fmt(&v, f));
    }
    Result::Ok(())
}

pub fn map_fmt_binary<K, V, I: Iterator<Item=(K, V)>>(s: I, f: &mut Formatter) -> fmt::Result
    where K:fmt::Binary,
          V:fmt::Binary
{
    for (i, (k, v)) in s.enumerate() {
        if i != 0 { try!(write!(f, ", ")); }
        try!(fmt::Binary::fmt(&k, f));
        try!(write!(f, ": "));
        try!(fmt::Binary::fmt(&v, f));
    }
    Result::Ok(())
}

pub fn map_fmt_upper_hex<K, V, I: Iterator<Item=(K, V)>>(s: I, f: &mut Formatter) -> fmt::Result
    where K:fmt::UpperHex,
          V:fmt::UpperHex
{
    for (i, (k, v)) in s.enumerate() {
        if i != 0 { try!(write!(f, ", ")); }
        try!(fmt::UpperHex::fmt(&k, f));
        try!(write!(f, ": "));
        try!(fmt::UpperHex::fmt(&v, f));
    }
    Result::Ok(())
}

pub fn map_fmt_lower_hex<K, V, I: Iterator<Item=(K, V)>>(s: I, f: &mut Formatter) -> fmt::Result
    where K:fmt::LowerHex,
          V:fmt::LowerHex
{
    for (i, (k, v)) in s.enumerate() {
        if i != 0 { try!(write!(f, ", ")); }
        try!(fmt::LowerHex::fmt(&k, f));
        try!(write!(f, ": "));
        try!(fmt::LowerHex::fmt(&v, f));
    }
    Result::Ok(())
}

pub fn map_fmt_upper_exp<K, V, I: Iterator<Item=(K, V)>>(s: I, f: &mut Formatter) -> fmt::Result
    where K:fmt::UpperExp,
          V:fmt::UpperExp
{
    for (i, (k, v)) in s.enumerate() {
        if i != 0 { try!(write!(f, ", ")); }
        try!(fmt::UpperExp::fmt(&k, f));
        try!(write!(f, ": "));
        try!(fmt::UpperExp::fmt(&v, f));
    }
    Result::Ok(())
}

pub fn map_fmt_lower_exp<K, V, I: Iterator<Item=(K, V)>>(s: I, f: &mut Formatter) -> fmt::Result
    where K:fmt::LowerExp,
          V:fmt::LowerExp
{
    for (i, (k, v)) in s.enumerate() {
        if i != 0 { try!(write!(f, ", ")); }
        try!(fmt::LowerExp::fmt(&k, f));
        try!(write!(f, ": "));
        try!(fmt::LowerExp::fmt(&v, f));
    }
    Result::Ok(())
}
