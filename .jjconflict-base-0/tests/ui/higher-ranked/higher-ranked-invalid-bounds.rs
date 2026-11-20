//! Tests invalid lifetime bounds and generic parameters in higher-ranked types.

type A = for<'b, 'a: 'b> fn(); //~ ERROR bounds cannot be used in this context
type B = for<'b, 'a: 'b,> fn(); //~ ERROR bounds cannot be used in this context
type C = for<'b, 'a: 'b +> fn(); //~ ERROR bounds cannot be used in this context
type D = for<'a, T> fn(); //~ ERROR only lifetime parameters can be used in this context
type E = dyn for<T, U> Fn(); //~ ERROR only lifetime parameters can be used in this context

fn main() {}
