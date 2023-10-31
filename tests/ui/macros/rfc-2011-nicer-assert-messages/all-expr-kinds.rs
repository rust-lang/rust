// edition:2021
// ignore-tidy-linelength
// only-x86_64
// run-pass
// needs-unwind Asserting on contents of error message

#![allow(path_statements, unused_allocation)]
#![feature(core_intrinsics, generic_assert)]

macro_rules! test {
  (
    let mut $elem_ident:ident = $elem_expr:expr;
    [ $($assert:tt)* ] => $msg:literal
  ) => {
    {
      #[allow(unused_assignments, unused_mut, unused_variables)]
      let rslt = std::panic::catch_unwind(|| {
        let mut $elem_ident = $elem_expr;
        assert!($($assert)*);
      });
      let err = rslt.unwrap_err();
      if let Some(elem) = err.downcast_ref::<String>() {
        assert_eq!(elem, &$msg);
      }
      else if let Some(elem) = err.downcast_ref::<&str>() {
        assert_eq!(elem, &$msg);
      }
      else {
        panic!("assert!( ... ) should return a string");
      }
    }
  }
}

macro_rules! tests {
  (
    let mut $elem_ident:ident = $elem_expr:expr;

    $(
      [ $($elem_assert:tt)* ] => $elem_msg:literal
    )+
  ) => {
    $(
      test!(
        let mut $elem_ident = $elem_expr;
        [ $($elem_assert)* ] => $elem_msg
      );
    )+
  }
}

const FOO: Foo = Foo { bar: 1 };


#[derive(Clone, Copy, Debug, PartialEq)]
struct Foo {
  bar: i32
}

impl Foo {
  fn add(&self, a: i32, b: i32) -> i32 { a + b }
}

fn add(a: i32, b: i32) -> i32 { a + b }

fn main() {
  // ***** Allowed *****

  tests!(
    let mut elem = 1i32;

    // addr of
    [ &elem == &3 ] => "Assertion failed: &elem == &3\nWith captures:\n  elem = 1\n"

    // array
    [ [elem][0] == 3 ] => "Assertion failed: [elem][0] == 3\nWith captures:\n  elem = 1\n"

    // binary
    [ elem + 1 == 3 ] => "Assertion failed: elem + 1 == 3\nWith captures:\n  elem = 1\n"

    // call
    [ add(elem, elem) == 3 ] => "Assertion failed: add(elem, elem) == 3\nWith captures:\n  elem = 1\n"

    // cast
    [ elem as i32 == 3 ] => "Assertion failed: elem as i32 == 3\nWith captures:\n  elem = 1\n"

    // if
    [ if elem == 3 { true } else { false } ] => "Assertion failed: if elem == 3 { true } else { false }\nWith captures:\n  elem = 1\n"

    // index
    [ [1i32, 1][elem as usize] == 3 ] => "Assertion failed: [1i32, 1][elem as usize] == 3\nWith captures:\n  elem = 1\n"

    // let
    [ if let 3 = elem { true } else { false } ] => "Assertion failed: if let 3 = elem { true } else { false }\nWith captures:\n  elem = 1\n"

    // match
    [ match elem { 3 => true, _ => false, } ] => "Assertion failed: match elem { 3 => true, _ => false, }\nWith captures:\n  elem = 1\n"

    // method call
    [ FOO.add(elem, elem) == 3 ] => "Assertion failed: FOO.add(elem, elem) == 3\nWith captures:\n  elem = 1\n"

    // paren
    [ (elem) == 3 ] => "Assertion failed: (elem) == 3\nWith captures:\n  elem = 1\n"

    // range
    [ (0..elem) == (0..3) ] => "Assertion failed: (0..elem) == (0..3)\nWith captures:\n  elem = 1\n"

    // repeat
    [ [elem; 1] == [3; 1] ] => "Assertion failed: [elem; 1] == [3; 1]\nWith captures:\n  elem = 1\n"

    // struct
    [ Foo { bar: elem } == Foo { bar: 3 } ] => "Assertion failed: Foo { bar: elem } == Foo { bar: 3 }\nWith captures:\n  elem = 1\n"

    // tuple
    [ (elem, 1) == (3, 3) ] => "Assertion failed: (elem, 1) == (3, 3)\nWith captures:\n  elem = 1\n"

    // unary
    [ -elem == -3 ] => "Assertion failed: -elem == -3\nWith captures:\n  elem = 1\n"
  );
}
