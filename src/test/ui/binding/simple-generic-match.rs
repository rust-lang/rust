// run-pass
#![allow(non_camel_case_types, type_param_on_variant_ctor)]

// pretty-expanded FIXME #23616

enum clam<T> { a(T), }

pub fn main() { let c = clam::a(2); match c { clam::a::<isize>(_) => { } } }
