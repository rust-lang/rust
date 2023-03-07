type A = for<'b, 'a: 'b> fn(); //~ ERROR lifetime bounds cannot be used in this context
type B = for<'b, 'a: 'b,> fn(); //~ ERROR lifetime bounds cannot be used in this context
type C = for<'b, 'a: 'b +> fn(); //~ ERROR lifetime bounds cannot be used in this context
type D = for<'a, T> fn(); //~ ERROR only lifetime parameters can be used in this context
type E = dyn for<T> Fn(); //~ ERROR only lifetime parameters can be used in this context

fn main() {}
