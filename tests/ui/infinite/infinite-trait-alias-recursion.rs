#![feature(trait_alias)]
//@ ignore-parallel-frontend query cycle
trait T1 = T2;
//~^ ERROR cycle detected when computing the implied predicates of `T1`

trait T2 = T3;

trait T3 = T1 + T3;

fn main() {}
