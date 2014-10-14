- Start Date: 2014-07-04
- RFC PR #: [rust-lang/rfcs#155](https://github.com/rust-lang/rfcs/pull/155)
- Rust Issue #: [rust-lang/rust#17059](https://github.com/rust-lang/rust/issues/17059)

# Summary

Require "anonymous traits", i.e. `impl MyStruct` to occur only in the same module that `MyStruct` is defined.

# Motivation

Before I can explain the motivation for this, I should provide some background
as to how anonymous traits are implemented, and the sorts of bugs we see with
the current behaviour. The conclusion will be that we effectively already only
support `impl MyStruct` in the same module that `MyStruct` is defined, and
making this a rule will simply give cleaner error messages.

- The compiler first sees `impl MyStruct` during the resolve phase, specifically
  in `Resolver::build_reduced_graph()`, called by `Resolver::resolve()` in
  `src/librustc/middle/resolve.rs`. This is before any type checking (or type
  resolution, for that matter) is done, so the compiler trusts for now that
  `MyStruct` is a valid type.
- If `MyStruct` is a path with more than one segment, such as `mymod::MyStruct`,
  it is silently ignored (how was this not flagged when the code was written??),
  which effectively causes static methods in such `impl`s to be dropped on the
  floor. A silver lining here is that nothing is added to the current module
  namespace, so the shadowing bugs demonstrated in the next bullet point do not
  apply here. (To locate this bug in the code, find the `match` immediately following
  the `FIXME (#3785)` comment in `resolve.rs`.) This leads to the following
````
mod break1 {
    pub struct MyGuy;

    impl MyGuy {
        pub fn do1() { println!("do 1"); }
    }
}

impl break1::MyGuy {
    fn do2() { println!("do 2"); }
}

fn main() {
    break1::MyGuy::do1();
    break1::MyGuy::do2();
}
````
````
<anon>:15:5: 15:23 error: unresolved name `break1::MyGuy::do2`.
<anon>:15     break1::MyGuy::do2();
````
  as noticed by @huonw in https://github.com/rust-lang/rust/issues/15060 .
- If one does not exist, the compiler creates a submodule `MyStruct` of the
  current module, with `kind` `ImplModuleKind`. Static methods are placed into
  this module. If such a module already exists, the methods are appended to it,
  to support multiple `impl MyStruct` blocks within the same module. If a module
  exists that is not `ImplModuleKind`, the compiler signals a duplicate module
  definition error.
- Notice at this point that if there is a `use MyStruct`, the compiler will act
  as though it is unaware of this. This is because imports are not resolved yet
  (they are in `Resolver::resolve_imports()` called immediately after
  `Resolver::build_reduced_graph()` is called). In the final resolution step,
  `MyStruct` will be searched in the namespace of the current module, checking
  imports only as a fallback (and only in some contexts), so the `use MyStruct` is
  effectively shadowed. If there is an `impl MyStruct` in the file being imported
  from, the user expects that the new `impl MyStruct` will append to that one,
  same as if they are in the original file. This leads to the original bug report
  https://github.com/rust-lang/rust/issues/15060 .
- In fact, even if no methods from the import are used, the name `MyStruct` will
  not be associated to a type, so that
````
trait T {}
impl<U: T> Vec<U> {
    fn from_slice<'a>(x: &'a [uint]) -> Vec<uint> {
        fail!()
    }
}
fn main() { let r = Vec::from_slice(&[1u]); }
````
````
error: found module name used as a type: impl Vec<U>::Vec<U> (id=5)
impl<U: T> Vec<U>
````
  which @Ryman noticed in https://github.com/rust-lang/rust/issues/15060 . The
  reason for this is that in `Resolver::resolve_crate()`, the final step of
  `Resolver::resolve()`, the type of an anonymous `impl` is determined by
  `NameBindings::def_for_namespace(TypeNS)`. This function searches the namespace
  `TypeNS` (which is <i>not</i> affected by imports) for a type; failing that it
  tries for a module; failing that it returns `None`. The result is that when
  typeck runs, it sees `impl [module name]` instead of `impl [type name]`.
  

The main motivation of this RFC is to clear out these bugs, which do not make
sense to a user of the language (and had me confused for quite a while).

A secondary motivation is to enforce consistency in code layout; anonymous traits
are used the way that class methods are used in other languages, and the data
and methods of a struct should be defined nearby.

# Detailed design

I propose three changes to the language:

- `impl` on multiple-ident paths such as `impl mymod::MyStruct` is disallowed.
  Since this currently suprises the user by having absolutely no effect for
  static methods, support for this is already broken.
- `impl MyStruct` must occur in the same module that `MyStruct` is defined.
  This is to prevent the above problems with `impl`-across-modules.
  Migration path is for users to just move code between source files.

# Drawbacks

Static methods on `impl`s-away-from-definition never worked, while non-static
methods can be implemented using non-anonymous traits. So there is no loss in
expressivity. However, using a trait where before there was none may be clumsy,
since it might not have a sensible name, and it must be explicitly imported by
all users of the trait methods.

For example, in the stdlib `src/libstd/io/fs.rs` we see the code `impl path::Path`
to attach (non-static) filesystem-related methods to the `Path` type. This would
have to be done via a `FsPath` trait which is implemented on `Path` and exported
alongside `Path` in the prelude.

It is worth noting that this is the only instance of this RFC conflicting with
current usage in the stdlib or compiler.

# Alternatives

- Leaving this alone and fixing the bugs directly. This is really hard. To do it
  properly, we would need to seriously refactor resolve.

# Unresolved questions

None.



