#![feature(default_type_parameter_fallback)]

fn avg<T=T::Item>(_: T) {}
//~^ ERROR type parameters with a default cannot use forward declared identifiers

fn main() {}
