% Coercions

Types can implicitly be coerced to change in certain contexts. These changes are
generally just *weakening* of types, largely focused around pointers and
lifetimes. They mostly exist to make Rust "just work" in more cases, and are
largely harmless.

Here's all the kinds of coercion:

Coercion is allowed between the following types:

* Transitivity: `T_1` to `T_3` where `T_1` coerces to `T_2` and `T_2` coerces to
  `T_3`
* Pointer Weakening:
    * `&mut T` to `&T`
    * `*mut T` to `*const T`
    * `&T` to `*const T`
    * `&mut T` to `*mut T`
* Unsizing: `T` to `U` if `T` implements `CoerceUnsized<U>`
* Deref coercion: Expression `&x` of type `&T` to `&*x` of type `&U` if `T` derefs to `U` (i.e. `T: Deref<Target=U>`)

`CoerceUnsized<Pointer<U>> for Pointer<T> where T: Unsize<U>` is implemented
for all pointer types (including smart pointers like Box and Rc). Unsize is
only implemented automatically, and enables the following transformations:

* `[T; n]` => `[T]`
* `T` => `Trait` where `T: Trait`
* `Foo<..., T, ...>` => `Foo<..., U, ...>` where:
    * `T: Unsize<U>`
    * `Foo` is a struct
    * Only the last field of `Foo` has type involving `T`
    * `T` is not part of the type of any other fields
    * `Bar<T>: Unsize<Bar<U>>`, if the last field of `Foo` has type `Bar<T>`

Coercions occur at a *coercion site*. Any location that is explicitly typed
will cause a coercion to its type. If inference is necessary, the coercion will
not be performed. Exhaustively, the coercion sites for an expression `e` to
type `U` are:

* let statements, statics, and consts: `let x: U = e`
* Arguments to functions: `takes_a_U(e)`
* Any expression that will be returned: `fn foo() -> U { e }`
* Struct literals: `Foo { some_u: e }`
* Array literals: `let x: [U; 10] = [e, ..]`
* Tuple literals: `let x: (U, ..) = (e, ..)`
* The last expression in a block: `let x: U = { ..; e }`

Note that we do not perform coercions when matching traits (except for
receivers, see below). If there is an impl for some type `U` and `T` coerces to
`U`, that does not constitute an implementation for `T`. For example, the
following will not type check, even though it is OK to coerce `t` to `&T` and
there is an impl for `&T`:

```rust,ignore
trait Trait {}

fn foo<X: Trait>(t: X) {}

impl<'a> Trait for &'a i32 {}


fn main() {
    let t: &mut i32 = &mut 0;
    foo(t);
}
```

```text
<anon>:10:5: 10:8 error: the trait bound `&mut i32 : Trait` is not satisfied [E0277]
<anon>:10     foo(t);
              ^~~
```
