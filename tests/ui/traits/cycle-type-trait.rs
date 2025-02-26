//@ run-pass
#![allow(dead_code)]
// Test a case where a supertrait references a type that references
// the original trait. This poses no problem at the moment.


trait Chromosome: Get<Struct<i32>> {
}

trait Get<A> {
    fn get(&self) -> A;
}

struct Struct<C:Chromosome> { c: C }

impl Chromosome for i32 { }

impl Get<Struct<i32>> for i32 {
    fn get(&self) -> Struct<i32> {
        Struct { c: *self }
    }
}

fn main() { }
