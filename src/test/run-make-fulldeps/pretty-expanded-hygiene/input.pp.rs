// minimal junk
#![feature(no_core)]
#![no_core]

macro_rules! foo /* 60#0 */(( $ x : ident ) => { y + $ x });

fn bar /* 62#0 */() { let x /* 59#2 */ = 1; y /* 61#4 */ + x /* 59#5 */ }

fn y /* 61#0 */() { }
