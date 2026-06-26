//@ edition:2018

#![feature(async_trait_bounds)]

struct S;

fn test(x: impl async S) {}
//~^ ERROR expected trait, found struct `S`

fn missing(x: impl async Missing) {}
//~^ ERROR cannot find trait `Missing` in this scope

fn main() {}
