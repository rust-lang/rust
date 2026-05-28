#![allow(dead_code)]
#![forbid(non_snake_case)]

// 2. non_snake_case


fn ǇǊaaǈǊaǄooo() {}
//~^ ERROR function `ǇǊaaǈǊaǄooo` should have a snake case name
//~| WARN identifier contains 5 non normalized (NFKC) characters

fn ǈǌaaǈǋaǅooo() {}
//~^ ERROR function `ǈǌaaǈǋaǅooo` should have a snake case name
//~| WARN identifier contains 5 non normalized (NFKC) characters

// test final sigma casing
fn ΦΙΛΟΣ_ΦΙΛΟΣ() {}
//~^ ERROR function `ΦΙΛΟΣ_ΦΙΛΟΣ` should have a snake case name

fn Σ() {}
//~^ ERROR function `Σ` should have a snake case name

fn ΦΙΛΟΣ_Σ() {}
//~^ ERROR function `ΦΙΛΟΣ_Σ` should have a snake case name

fn Σ_ΦΙΛΟΣ() {}
//~^ ERROR function `Σ_ΦΙΛΟΣ` should have a snake case name

// this is ok
fn φιλοσ_φιλοσ() {}

fn main() {}
