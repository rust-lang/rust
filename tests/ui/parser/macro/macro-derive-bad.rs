#![crate_type = "lib"]
#![feature(macro_derive)]

macro_rules! derive_incomplete_1 { derive }
//~^ ERROR macro definition ended unexpectedly
//~| NOTE expected `()` after `derive`

macro_rules! derive_incomplete_2 { derive() }
//~^ ERROR macro definition ended unexpectedly
//~| NOTE expected macro derive body

macro_rules! derive_incomplete_3 { derive() {} }
//~^ ERROR expected `=>`
//~| NOTE expected `=>`

macro_rules! derive_incomplete_4 { derive() {} => }
//~^ ERROR macro definition ended unexpectedly
//~| NOTE expected right-hand side of macro rule

macro_rules! derive_noparens_1 { derive{} {} => {} }
//~^ ERROR `derive` rule argument matchers require parentheses

macro_rules! derive_noparens_2 { derive[] {} => {} }
//~^ ERROR `derive` rule argument matchers require parentheses

macro_rules! derive_noparens_3 { derive _ {} => {} }
//~^ ERROR `derive` must be followed by `()`

macro_rules! derive_args_1 { derive($x:ident) ($y:ident) => {} }
//~^ ERROR `derive` rules do not accept arguments

macro_rules! derive_args_2 { derive() => {} }
//~^ ERROR expected macro derive body, got `=>`

macro_rules! derive_args_3 { derive($x:ident) => {} }
//~^ ERROR `derive` rules do not accept arguments
//~| ERROR expected macro derive body, got `=>`
//~| NOTE need `()` after this `derive`

macro_rules! derive_dup_matcher { derive() {$x:ident $x:ident} => {} }
//~^ ERROR duplicate matcher binding
//~| NOTE duplicate binding
//~| NOTE previous binding

macro_rules! derive_unsafe { unsafe derive() {} => {} }
//~^ ERROR `unsafe` is only supported on `attr` rules
