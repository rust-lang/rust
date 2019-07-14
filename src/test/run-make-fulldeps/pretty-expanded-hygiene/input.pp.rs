// minimal junk
#![feature /* 0#0 */(no_core)]
#![no_core /* 0#0 */]

macro_rules! foo /* 0#0 */ { ($ x : ident) => { y + $ x } }

fn bar /* 0#0 */() { let x /* 0#0 */ = 1; y /* 0#1 */ + x /* 0#0 */ }

fn y /* 0#0 */() { }
