#![crate_type = "lib"]

macro_rules! a { {} => }
//~^ ERROR: macro definition ended unexpectedly

macro_rules! b { 0 => }
//~^ ERROR: macro definition ended unexpectedly
//~| ERROR: invalid macro matcher

macro_rules! c { x => }
//~^ ERROR: macro definition ended unexpectedly
//~| ERROR: invalid macro matcher

macro_rules! d { _ => }
//~^ ERROR: macro definition ended unexpectedly
//~| ERROR: invalid macro matcher

macro_rules! e { {} }
//~^ ERROR: expected `=>`, found end of macro arguments

macro_rules! f {}
//~^ ERROR: macros must contain at least one rule

macro_rules! g { unsafe {} => {} }
//~^ ERROR: `unsafe` is only supported on `attr` rules
