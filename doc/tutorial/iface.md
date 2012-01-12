# Interfaces

Interfaces are Rust's take on value polymorphism—the thing that
object-oriented languages tend to solve with methods and inheritance.
For example, writing a function that can operate on multiple types of
collections.

NOTE: This feature is very new, and will need a few extensions to be
applicable to more advanced use cases.

## Declaration

An interface consists of a set of methods. A method is a function that
can be applied to a `self` value and a number of arguments, using the
dot notation: `self.foo(arg1, arg2)`.

For example, we could declare the interface `to_str` for things that
can be converted to a string, with a single method of the same name:

    iface to_str {
        fn to_str() -> str;
    }

## Implementation

To actually implement an interface for a given type, the `impl` form
is used. This defines implementations of `to_str` for the `int` and
`str` types.

    # iface to_str { fn to_str() -> str; }
    impl of to_str for int {
        fn to_str() -> str { int::to_str(self, 10u) }
    }
    impl of to_str for str {
        fn to_str() -> str { self }
    }

Given these, we may call `1.to_str()` to get `"1"`, or
`"foo".to_str()` to get `"foo"` again. This is basically a form of
static overloading—when the Rust compiler sees the `to_str` method
call, it looks for an implementation that matches the type with a
method that matches the name, and simply calls that.

## Scoping

Implementations are not globally visible. Resolving a method to an
implementation requires that implementation to be in scope. You can
import and export implementations using the name of the interface they
implement (multiple implementations with the same name can be in scope
without problems). Or you can give them an explicit name if you
prefer, using this syntax:

    # iface to_str { fn to_str() -> str; }
    impl nil_to_str of to_str for () {
        fn to_str() -> str { "()" }
    }

## Bounded type parameters

The useful thing about value polymorphism is that it does not have to
be static. If object-oriented languages only let you call a method on
an object when they knew exactly which sub-type it had, that would not
get you very far. To be able to call methods on types that aren't
known at compile time, it is possible to specify 'bounds' for type
parameters.

    # iface to_str { fn to_str() -> str; }
    fn comma_sep<T: to_str>(elts: [T]) -> str {
        let result = "", first = true;
        for elt in elts {
            if first { first = false; }
            else { result += ", "; }
            result += elt.to_str();
        }
        ret result;
    }

The syntax for this is similar to the syntax for specifying that a
parameter type has to be copyable (which is, in principle, another
kind of bound). By declaring `T` as conforming to the `to_str`
interface, it becomes possible to call methods from that interface on
values of that type inside the function. It will also cause a
compile-time error when anyone tries to call `comma_sep` on an array
whose element type does not have a `to_str` implementation in scope.

## Polymorphic interfaces

Interfaces may contain type parameters. This defines an interface for
generalized sequence types:

    iface seq<T> {
        fn len() -> uint;
        fn iter(block(T));
    }
    impl <T> of seq<T> for [T] {
        fn len() -> uint { vec::len(self) }
        fn iter(b: block(T)) {
            for elt in self { b(elt); }
        }
    }

Note that the implementation has to explicitly declare the its
parameter `T` before using it to specify its interface type. This is
needed because it could also, for example, specify an implementation
of `seq<int>`—the `of` clause *refers* to a type, rather than defining
one.

## Casting to an interface type

The above allows us to define functions that polymorphically act on
values of *an* unknown type that conforms to a given interface.
However, consider this function:

    # iface drawable { fn draw(); }
    fn draw_all<T: drawable>(shapes: [T]) {
        for shape in shapes { shape.draw(); }
    }

You can call that on an array of circles, or an array of squares
(assuming those have suitable `drawable` interfaces defined), but not
on an array containing both circles and squares.

When this is needed, an interface name can be used as a type, causing
the function to be written simply like this:

    # iface drawable { fn draw(); }
    fn draw_all(shapes: [drawable]) {
        for shape in shapes { shape.draw(); }
    }

There is no type parameter anymore (since there isn't a single type
that we're calling the function on). Instead, the `drawable` type is
used to refer to a type that is a reference-counted box containing a
value for which a `drawable` implementation exists, combined with
information on where to find the methods for this implementation. This
is very similar to the 'vtables' used in most object-oriented
languages.

To construct such a value, you use the `as` operator to cast a value
to an interface type:

    # type circle = int; type rectangle = int;
    # iface drawable { fn draw(); }
    # impl of drawable for int { fn draw() {} }
    # fn new_circle() -> int { 1 }
    # fn new_rectangle() -> int { 2 }
    # fn draw_all(shapes: [drawable]) {}
    let c: circle = new_circle();
    let r: rectangle = new_rectangle();
    draw_all([c as drawable, r as drawable]);

This will store the value into a box, along with information about the
implementation (which is looked up in the scope of the cast). The
`drawable` type simply refers to such boxes, and calling methods on it
always works, no matter what implementations are in scope.

Note that the allocation of a box is somewhat more expensive than
simply using a type parameter and passing in the value as-is, and much
more expensive than statically resolved method calls.

## Interface-less implementations

If you only intend to use an implementation for static overloading,
and there is no interface available that it conforms to, you are free
to leave off the `of` clause.

    # type currency = ();
    # fn mk_currency(x: int, s: str) {}
    impl int_util for int {
        fn times(b: block(int)) {
            let i = 0;
            while i < self { b(i); i += 1; }
        }
        fn dollars() -> currency {
            mk_currency(self, "USD")
        }
    }

This allows cutesy things like `send_payment(10.dollars())`. And the
nice thing is that it's fully scoped, so the uneasy feeling that
anybody with experience in object-oriented languages (with the
possible exception of Rubyists) gets at the sight of such things is
not justified. It's harmless!
