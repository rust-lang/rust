// Here we test for coercions in struct fields of nested type ascriptions
// inside a tuple using an unsized coercion and a coercion from &mut -> &

// run-pass

#![feature(type_ascription)]

use std::any::type_name;
use std::assert_eq;

fn type_of<T>(_: T) -> &'static str {
    type_name::<T>()
}

struct Foo<'a, 'b, T1, T2> {
  _a : (T1, (T1, &'a [T1]), &'b T2),
}

struct Bar {
  _b : u32,
}

fn main() {
  let mut bar = Bar {_b : 26};
  let arr = [4,5,6];
  let tup = (9, (3, &arr : &[u32]), &mut bar);
  assert_eq!(type_of(tup), "(i32, (i32, &[u32]), &mut coerce_struct_fields_pass_in_let_stmt::Bar)");
  let _ = Foo { _a : (9, (3, &arr : &[u32]), &mut bar) };
}
