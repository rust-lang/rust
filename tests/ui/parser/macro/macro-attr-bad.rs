#![crate_type = "lib"]
#![feature(macro_attr)]

macro_rules! attr_incomplete_1 { attr }
//~^ ERROR macro definition ended unexpectedly

macro_rules! attr_incomplete_2 { attr() }
//~^ ERROR macro definition ended unexpectedly

macro_rules! attr_incomplete_3 { attr() {} }
//~^ ERROR expected `=>`

macro_rules! attr_incomplete_4 { attr() {} => }
//~^ ERROR macro definition ended unexpectedly

macro_rules! attr_incomplete_5 { unsafe }
//~^ ERROR macro definition ended unexpectedly

macro_rules! attr_incomplete_6 { unsafe attr }
//~^ ERROR macro definition ended unexpectedly

macro_rules! attr_noparens_1 { attr{} {} => {} }
//~^ ERROR `attr` rule argument matchers require parentheses

macro_rules! attr_noparens_2 { attr[] {} => {} }
//~^ ERROR `attr` rule argument matchers require parentheses

macro_rules! attr_noparens_3 { attr _ {} => {} }
//~^ ERROR invalid macro matcher

macro_rules! attr_dup_matcher_1 { attr() {$x:ident $x:ident} => {} }
//~^ ERROR duplicate matcher binding

macro_rules! attr_dup_matcher_2 { attr($x:ident $x:ident) {} => {} }
//~^ ERROR duplicate matcher binding

macro_rules! attr_dup_matcher_3 { attr($x:ident) {$x:ident} => {} }
//~^ ERROR duplicate matcher binding
