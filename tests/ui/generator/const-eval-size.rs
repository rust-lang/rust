// error-pattern: evaluation of constant value failed

#![feature(generators)]
#![feature(generator_trait)]
#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

type Gen = impl std::ops::Generator;

const A: usize = std::mem::size_of::<Gen>();

const B: usize = std::mem::size_of::<Option<Gen>>();

fn gen() -> Gen {
    move || {
        yield;
    }
}

fn main() {}
