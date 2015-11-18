// rustfmt-wrap_comments: true
// Test of lots of random stuff.
// FIXME split this into multiple, self-contained tests.


#[attr1]
extern crate foo;
#[attr2]
#[attr3]
extern crate foo;
#[attr1]
extern crate foo;
#[attr2]
#[attr3]
extern crate foo;

use std::cell::*;
use std::{self, any, ascii, borrow, boxed, char, borrow, boxed, char, borrow, borrow, boxed, char,
          borrow, boxed, char, borrow, boxed, char, borrow, boxed, char, borrow, boxed, char,
          borrow, boxed, char, borrow, boxed, char, borrow, boxed, char};

mod doc;
mod other;


// sfdgfffffffffffffffffffffffffffffffffffffffffffffffffffffff
// ffffffffffffffffffffffffffffffffffffffffff

fn foo(a: isize, b: u32 /* blah blah */, c: f64) { }

fn foo() -> Box<Write + 'static>
    where 'a: 'b,
          for<'a> D<'b>: 'a
{
    hello!()
}

fn baz<'a: 'b, // comment on 'a
       T: SomsssssssssssssssssssssssssssssssssssssssssssssssssssssseType /* comment on T */>
    (a: A,
     b: B, // comment on b
     c: C)
     -> Bob {
    #[attr1]
    extern crate foo;
    #[attr2]
    #[attr3]
    extern crate foo;
    #[attr1]
    extern crate foo;
    #[attr2]
    #[attr3]
    extern crate foo;
}

#[rustfmt_skip]
fn qux(a: dadsfa,   // Comment 1
       b: sdfasdfa, // Comment 2
       c: dsfdsafa) // Comment 3
{

}

/// Blah blah blah.
impl Bar {
    fn foo(&mut self,
           a: sdfsdfcccccccccccccccccccccccccccccccccccccccccccccccccc, // comment on a
           b: sdfasdfsdfasfs /* closing comment */)
           -> isize {
    }

    /// Blah blah blah.
    pub fn f2(self) {
        (foo, bar)
    }

    #[an_attribute]
    fn f3(self) -> Dog { }
}

/// The `nodes` and `edges` method each return instantiations of
/// `Cow<[T]>` to leave implementers the freedom to create

/// entirely new vectors or to pass back slices into internally owned
/// vectors.
pub trait GraphWalk<'a, N, E> {
    /// Returns all the nodes in this graph.
    fn nodes(&'a self) -> Nodes<'a, N>;
    /// Returns all of the edges in this graph.
    fn edges(&'a self) -> Edges<'a, E>;
    /// The source node for `edge`.
    fn source(&'a self, edge: &E) -> N;
    /// The target node for `edge`.
    fn target(&'a self, edge: &E) -> N;
}

/// A Doc comment
#[AnAttribute]
pub struct Foo {
    #[rustfmt_skip]
    f :   SomeType, // Comment beside a field
    f: SomeType, // Comment beside a field
    // Comment on a field
    g: SomeOtherType,
    /// A doc comment on a field
    h: AThirdType,
}

struct Bar;

// With a where clause and generics.
pub struct Foo<'a, Y: Baz>
    where X: Whatever
{
    f: SomeType, // Comment beside a field
}

fn foo(ann: &'a (PpAnn + 'a)) { }

fn main() {
    for i in 0i32..4 {
        println!("{}", i);
    }


    while true {
        hello();
    }

    let rc = Cell::new(42usize,
                       42usize,
                       Cell::new(42usize,
                                 remaining_widthremaining_widthremaining_widthremaining_width),
                       42usize);
    let rc = RefCell::new(42usize, remaining_width, remaining_width);  // a comment
    let x = "Hello!!!!!!!!! abcd  abcd abcd abcd abcd abcd\n abcd abcd abcd abcd abcd abcd abcd \
             abcd abcd abcd  abcd abcd abcd abcd abcd abcd abcd abcd abcd abcd abcd abcd abcd \
             abcd abcd abcd abcd abcd abcd abcd abcd abcd abcd abcd abcd abcd abcd abcd abcd abcd \
             abcd abcd";
    let s = expand(a, b);
}

fn deconstruct()
    -> (SocketAddr,
        Method,
        Headers,
        RequestUri,
        HttpVersion,
        AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA)
{
}

fn deconstruct(foo: Bar)
               -> (SocketAddr,
                   Method,
                   Headers,
                   RequestUri,
                   HttpVersion,
                   AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA) {
}
