#![feature(mem_conjure_zst)]

use std::{convert::Infallible, mem};

const INVALID: Infallible = unsafe { mem::conjure_zst() };
//~^ ERROR attempted to instantiate uninhabited type

const VALID: () = unsafe { mem::conjure_zst() };

fn main() {}
