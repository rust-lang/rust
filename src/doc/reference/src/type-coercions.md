# Type coercions

Coercions are defined in [RFC 401]. A coercion is implicit and has no syntax.

[RFC 401]: https://github.com/rust-lang/rfcs/blob/master/text/0401-coercions.md

## Coercion sites

A coercion can only occur at certain coercion sites in a program; these are
typically places where the desired type is explicit or can be derived by
propagation from explicit types (without type inference). Possible coercion
sites are:

* `let` statements where an explicit type is given.

   For example, `42` is coerced to have type `i8` in the following:

   ```rust
   let _: i8 = 42;
   ```

* `static` and `const` statements (similar to `let` statements).

* Arguments for function calls

  The value being coerced is the actual parameter, and it is coerced to
  the type of the formal parameter.

  For example, `42` is coerced to have type `i8` in the following:

  ```rust
  fn bar(_: i8) { }

  fn main() {
      bar(42);
  }
  ```

* Instantiations of struct or variant fields

  For example, `42` is coerced to have type `i8` in the following:

  ```rust
  struct Foo { x: i8 }

  fn main() {
      Foo { x: 42 };
  }
  ```

* Function results, either the final line of a block if it is not
  semicolon-terminated or any expression in a `return` statement

  For example, `42` is coerced to have type `i8` in the following:

  ```rust
  fn foo() -> i8 {
      42
  }
  ```

If the expression in one of these coercion sites is a coercion-propagating
expression, then the relevant sub-expressions in that expression are also
coercion sites. Propagation recurses from these new coercion sites.
Propagating expressions and their relevant sub-expressions are:

* Array literals, where the array has type `[U; n]`. Each sub-expression in
the array literal is a coercion site for coercion to type `U`.

* Array literals with repeating syntax, where the array has type `[U; n]`. The
repeated sub-expression is a coercion site for coercion to type `U`.

* Tuples, where a tuple is a coercion site to type `(U_0, U_1, ..., U_n)`.
Each sub-expression is a coercion site to the respective type, e.g. the
zeroth sub-expression is a coercion site to type `U_0`.

* Parenthesized sub-expressions (`(e)`): if the expression has type `U`, then
the sub-expression is a coercion site to `U`.

* Blocks: if a block has type `U`, then the last expression in the block (if
it is not semicolon-terminated) is a coercion site to `U`. This includes
blocks which are part of control flow statements, such as `if`/`else`, if
the block has a known type.

## Coercion types

Coercion is allowed between the following types:

* `T` to `U` if `T` is a subtype of `U` (*reflexive case*)

* `T_1` to `T_3` where `T_1` coerces to `T_2` and `T_2` coerces to `T_3`
(*transitive case*)

    Note that this is not fully supported yet

* `&mut T` to `&T`

* `*mut T` to `*const T`

* `&T` to `*const T`

* `&mut T` to `*mut T`

* `&T` to `&U` if `T` implements `Deref<Target = U>`. For example:

  ```rust
  use std::ops::Deref;

  struct CharContainer {
      value: char,
  }

  impl Deref for CharContainer {
      type Target = char;

      fn deref<'a>(&'a self) -> &'a char {
          &self.value
      }
  }

  fn foo(arg: &char) {}

  fn main() {
      let x = &mut CharContainer { value: 'y' };
      foo(x); //&mut CharContainer is coerced to &char.
  }
  ```

* `&mut T` to `&mut U` if `T` implements `DerefMut<Target = U>`.

* TyCtor(`T`) to TyCtor(coerce_inner(`T`)), where TyCtor(`T`) is one of
    - `&T`
    - `&mut T`
    - `*const T`
    - `*mut T`
    - `Box<T>`

    and where
    - coerce_inner(`[T, ..n]`) = `[T]`
    - coerce_inner(`T`) = `U` where `T` is a concrete type which implements the
    trait `U`.

    In the future, coerce_inner will be recursively extended to tuples and
    structs. In addition, coercions from sub-traits to super-traits will be
    added. See [RFC 401] for more details.
