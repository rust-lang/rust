// Copyright (c) 2018 Nuxi (https://nuxi.nl/) and contributors.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

// Appease Rust's tidy.
// ignore-license

#[cfg(feature = "bitflags")]
#[macro_use]
extern crate bitflags;

// Minimal implementation of bitflags! in case we can't depend on the bitflags
// crate. Only implements `bits()` and a `from_bits_truncate()` that doesn't
// actually truncate.
#[cfg(not(feature = "bitflags"))]
macro_rules! bitflags {
  (
    $(#[$attr:meta])*
    pub struct $name:ident: $type:ty {
      $($(#[$const_attr:meta])* const $const:ident = $val:expr;)*
    }
  ) => {
    $(#[$attr])*
    #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
    pub struct $name { bits: $type }
    impl $name {
      $($(#[$const_attr])* pub const $const: $name = $name{ bits: $val };)*
      pub fn bits(&self) -> $type { self.bits }
      pub fn from_bits_truncate(bits: $type) -> Self { $name{ bits } }
    }
  }
}
