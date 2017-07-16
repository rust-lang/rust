// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


pub trait Foo {
    // @has assoc_consts/trait.Foo.html '//*[@class="rust trait"]' \
    //      'const FOO: usize;'
    // @has - '//*[@id="associatedconstant.FOO"]' 'const FOO: usize'
    // @has - '//*[@class="docblock"]' 'FOO: usize = 12'
    const FOO: usize = 12;
    // @has - '//*[@id="associatedconstant.FOO_NO_DEFAULT"]' 'const FOO_NO_DEFAULT: bool'
    const FOO_NO_DEFAULT: bool;
    // @!has - FOO_HIDDEN
    #[doc(hidden)]
    const FOO_HIDDEN: u8 = 0;
}

pub struct Bar;

impl Foo for Bar {
    // @has assoc_consts/struct.Bar.html '//code' 'impl Foo for Bar'
    // @has - '//*[@id="associatedconstant.FOO"]' 'const FOO: usize'
    // @has - '//*[@class="docblock"]' 'FOO: usize = 12'
    const FOO: usize = 12;
    // @has - '//*[@id="associatedconstant.FOO_NO_DEFAULT"]' 'const FOO_NO_DEFAULT: bool'
    // @has - '//*[@class="docblock"]' 'FOO_NO_DEFAULT: bool = false'
    const FOO_NO_DEFAULT: bool = false;
    // @!has - FOO_HIDDEN
    #[doc(hidden)]
    const FOO_HIDDEN: u8 = 0;
}

impl Bar {
    // @has assoc_consts/struct.Bar.html '//*[@id="associatedconstant.BAR"]' \
    //      'const BAR: usize'
    // @has - '//*[@class="docblock"]' 'BAR: usize = 3'
    pub const BAR: usize = 3;
}

pub struct Baz<'a, U: 'a, T>(T, &'a [U]);

impl Bar {
    // @has assoc_consts/struct.Bar.html '//*[@id="associatedconstant.BAZ"]' \
    //      "const BAZ: Baz<'static, u8, u32>"
    // @has - '//*[@class="docblock"]' "BAZ: Baz<'static, u8, u32> = Baz(321, &[1, 2, 3])"
    pub const BAZ: Baz<'static, u8, u32> = Baz(321, &[1, 2, 3]);
}

pub fn f(_: &(ToString + 'static)) {}

impl Bar {
    // @has assoc_consts/struct.Bar.html '//*[@id="associatedconstant.F"]' \
    //      "const F: fn(_: &(ToString + 'static))"
    // @has - '//*[@class="docblock"]' "F: fn(_: &(ToString + 'static)) = f"
    pub const F: fn(_: &(ToString + 'static)) = f;
}

impl Bar {
    // @!has assoc_consts/struct.Bar.html 'BAR_PRIVATE'
    const BAR_PRIVATE: char = 'a';
    // @!has assoc_consts/struct.Bar.html 'BAR_HIDDEN'
    #[doc(hidden)]
    pub const BAR_HIDDEN: &'static str = "a";
}

// @has assoc_consts/trait.Qux.html
pub trait Qux {
    // @has - '//*[@id="associatedconstant.QUX0"]' 'const QUX0: u8'
    // @has - '//*[@class="docblock"]' "Docs for QUX0 in trait."
    /// Docs for QUX0 in trait.
    const QUX0: u8;
    // @has - '//*[@id="associatedconstant.QUX1"]' 'const QUX1: i8'
    // @has - '//*[@class="docblock"]' "Docs for QUX1 in trait."
    /// Docs for QUX1 in trait.
    const QUX1: i8;
    // @has - '//*[@id="associatedconstant.QUX_DEFAULT0"]' 'const QUX_DEFAULT0: u16'
    // @has - '//*[@class="docblock"]' "QUX_DEFAULT0: u16 = 1"
    // @has - '//*[@class="docblock"]' "Docs for QUX_DEFAULT0 in trait."
    /// Docs for QUX_DEFAULT0 in trait.
    const QUX_DEFAULT0: u16 = 1;
    // @has - '//*[@id="associatedconstant.QUX_DEFAULT1"]' 'const QUX_DEFAULT1: i16'
    // @has - '//*[@class="docblock"]' "QUX_DEFAULT1: i16 = 2"
    // @has - '//*[@class="docblock"]' "Docs for QUX_DEFAULT1 in trait."
    /// Docs for QUX_DEFAULT1 in trait.
    const QUX_DEFAULT1: i16 = 2;
    // @has - '//*[@id="associatedconstant.QUX_DEFAULT2"]' 'const QUX_DEFAULT2: u32'
    // @has - '//*[@class="docblock"]' "QUX_DEFAULT2: u32 = 3"
    // @has - '//*[@class="docblock"]' "Docs for QUX_DEFAULT2 in trait."
    /// Docs for QUX_DEFAULT2 in trait.
    const QUX_DEFAULT2: u32 = 3;
}

// @has assoc_consts/struct.Bar.html '//code' 'impl Qux for Bar'
impl Qux for Bar {
    // @has - '//*[@id="associatedconstant.QUX0"]' 'const QUX0: u8'
    // @has - '//*[@class="docblock"]' "QUX0: u8 = 4"
    // @has - '//*[@class="docblock"]' "Docs for QUX0 in trait."
    /// Docs for QUX0 in trait.
    const QUX0: u8 = 4;
    // @has - '//*[@id="associatedconstant.QUX1"]' 'const QUX1: i8'
    // @has - '//*[@class="docblock"]' "QUX1: i8 = 5"
    // @has - '//*[@class="docblock"]' "Docs for QUX1 in impl."
    /// Docs for QUX1 in impl.
    const QUX1: i8 = 5;
    // @has - '//*[@id="associatedconstant.QUX_DEFAULT0"]' 'const QUX_DEFAULT0: u16'
    // @has - '//*[@class="docblock"]' "QUX_DEFAULT0: u16 = 6"
    // @has - '//*[@class="docblock"]' "Docs for QUX_DEFAULT0 in trait."
    const QUX_DEFAULT0: u16 = 6;
    // @has - '//*[@id="associatedconstant.QUX_DEFAULT1"]' 'const QUX_DEFAULT1: i16'
    // @has - '//*[@class="docblock"]' "QUX_DEFAULT1: i16 = 7"
    // @has - '//*[@class="docblock"]' "Docs for QUX_DEFAULT1 in impl."
    /// Docs for QUX_DEFAULT1 in impl.
    const QUX_DEFAULT1: i16 = 7;
    // @has - '//*[@id="associatedconstant.QUX_DEFAULT2"]' 'const QUX_DEFAULT2: u32'
    // @has - '//*[@class="docblock"]' "QUX_DEFAULT2: u32 = 3"
    // @has - '//*[@class="docblock"]' "Docs for QUX_DEFAULT2 in trait."
}
