# Monomorphization

As you probably know, rust has a very expressive type system that has extensive
support for generic types. But of course, assembly is not generic, so we need
to figure out the concrete types of all the generics before the code can
execute.

Different languages handle this problem differently. For example, in some
languages, such as Java, we may not know the most precise type of value until
runtime. In the case of Java, this is ok because (almost) all variables are
reference values anyway (i.e. pointers to a stack allocated object). This
flexibility comes at the cost of performance, since all accesses to an object
must dereference a pointer.

Rust takes a different approach: it _monomorphizes_ all generic types. This
means that compiler stamps out a different copy of the code of a generic
function for each concrete type needed. For example, if I use a `Vec<u64>` and
a `Vec<String>` in my code, then the generated binary will have two copies of
the generated code for `Vec`: one for `Vec<u64>` and another for `Vec<String>`.
The result is fast programs, but it comes at the cost of compile time (creating
all those copies can take a while) and binary size (all those copies might take
a lot of space).

Monomorphization is the first step in the backend of the rust compiler.

## Collection

First, we need to figure out what concrete types we need for all the generic
things in our program. This is called _collection_, and the code that does this
is called the _monomorphization collector_.

Take this example:

```rust
fn banana() {
   peach::<u64>();
}

fn main() {
    banana();
}
```

The monomorphization collector will give you a list of `[main, banana,
peach::<u64>]`. These are the functions that will have machine code generated
for them. Collector will also add things like statics to that list.

See [the collector rustdocs][collect] for more info.

[collect]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/monomorphize/collector/index.html

The monomorphization collector is run just before MIR lowering and codegen.
[`rustc_codegen_ssa::base::codegen_crate`][codegen1] calls the
[`collect_and_partition_mono_items`][mono] query, which does monomorphization
collection and then partitions them into [codegen
units](../appendix/glossary.md#codegen-unit).

[mono]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/monomorphize/partitioning/fn.collect_and_partition_mono_items.html
[codegen1]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_ssa/base/fn.codegen_crate.html

## Polymorphization

As mentioned above, monomorphization produces fast code, but it comes at the
cost of compile time and binary size. [MIR
optimizations](../mir/optimizations.md) can help a bit with this. Another
optimization currently under development is called _polymorphization_.

The general idea is that often we can share some code between monomorphized
copies of code. More precisely, if a MIR block is not dependent on a type
parameter, it may not need to be monomorphized into many copies. Consider the
following example:

```rust
pub fn f() {
    g::<bool>();
    g::<usize>();
}

fn g<T>() -> usize {
    let n = 1;
    let closure = || n;
    closure()
}
```

In this case, we would currently collect `[f, g::<bool>, g::<usize>,
g::<bool>::{{closure}}, g::<usize>::{{closure}}]`, but notice that the two
closures would be identical -- they don't depend on the type parameter `T` of
function `g`. So we only need to emit one copy of the closure.

For more information, see [this thread on github][polymorph].

[polymorph]: https://github.com/rust-lang/rust/issues/46477
