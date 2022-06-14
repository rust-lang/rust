// edition:2021
// ignore-tidy-linelength
// only-x86_64
// run-pass

#![allow(path_statements, unused_allocation)]
#![feature(box_syntax, core_intrinsics, generic_assert, generic_assert_internals)]

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

fn main() {
  // ***** Allowed *****

  tests!(
    let mut elem = 1i32;

    // binary
    [ elem + 1 == 3 ] => "Assertion failed: elem + 1 == 3\nWith captures:\n  elem = 1\n"
  );

  // ***** Disallowed *****

  tests!(
    let mut elem = 1i32;

    // assign
    [ { let local = elem; local } == 3 ] => "Assertion failed: { let local = elem; local } == 3"

    // assign op
    [ { elem += 1; elem } == 3 ] => "Assertion failed: { elem += 1; elem } == 3"

    // async
    [ { let _ = async { elem }; elem } == 3 ] => "Assertion failed: { let _ = async { elem }; elem } == 3"

    // await

    // block
    [ { elem } == 3 ] => "Assertion failed: { elem } == 3"

    // box
    [ box elem == box 3 ] => "Assertion failed: box elem == box 3"

    // break
    [ loop { break elem; } ==  3 ] => "Assertion failed: loop { break elem; } == 3"

    // closure
    [(|| elem)() ==  3 ] => "Assertion failed: (|| elem)() == 3"

    // const block

    // continue

    // err

    // field
    [ FOO.bar ==  3 ] => "Assertion failed: FOO.bar == 3"

    // for loop
    [ { for _ in 0..elem { elem; } elem } ==  3 ] => "Assertion failed: { for _ in 0..elem { elem; } elem } == 3"

    // if
    [ if true { elem } else { elem } == 3 ] => "Assertion failed: if true { elem } else { elem } == 3"

    // inline asm

    // let
    [ if let true = true { elem } else { elem } == 3 ] => "Assertion failed: if let true = true { elem } else { elem } == 3"

    // lit

    // loop
    [ loop { elem; break elem; } == 3 ] => "Assertion failed: loop { elem; break elem; } == 3"

    // mac call

    // match
    [ match elem { _ => elem } == 3 ] => "Assertion failed: match elem { _ => elem, } == 3"

    // ret
    [ (|| { return elem; })() == 3 ] => "Assertion failed: (|| { return elem; })() == 3"

    // try
    [ (|| { Some(Some(elem)?) })() == Some(3) ] => "Assertion failed: (|| { Some(Some(elem)?) })() == Some(3)"

    // try block

    // underscore

    // while
    [ { while false { elem; break; } elem } == 3 ] => "Assertion failed: { while false { elem; break; } elem } == 3"

    // yeet

    // yield
  );
}
