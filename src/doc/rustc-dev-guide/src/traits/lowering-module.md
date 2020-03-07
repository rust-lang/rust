# The lowering module in rustc

The program clauses described in the
[lowering rules](./lowering-rules.html) section are actually
created in the [`rustc_traits::lowering`][lowering] module.

[lowering]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_traits/lowering/

## The `program_clauses_for` query

The main entry point is the `program_clauses_for` [query], which –
given a `DefId` – produces a set of Chalk program clauses. The
query is invoked on a `DefId` that identifies something like a trait,
an impl, or an associated item definition. It then produces and
returns a vector of program clauses.

[query]: ../query.html

## Unit tests

**Note: We've removed the Chalk unit tests in [rust-lang/rust#69247].
They will come back once we're ready to integrate next Chalk into rustc.**

Here's a good example test. At the time of
this writing, it looked like this:

```rust,ignore
#![feature(rustc_attrs)]

trait Foo { }

#[rustc_dump_program_clauses] //~ ERROR program clause dump
impl<T: 'static> Foo for T where T: Iterator<Item = i32> { }

fn main() {
    println!("hello");
}
```

The `#[rustc_dump_program_clauses]` annotation can be attached to
anything with a `DefId` (It requires the `rustc_attrs` feature). The
compiler will then invoke the `program_clauses_for` query on that
item, and emit compiler errors that dump the clauses produced. These
errors just exist for unit-testing. The stderr will be:

```text
error: program clause dump
  --> $DIR/lower_impl.rs:5:1
   |
LL | #[rustc_dump_program_clauses]
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   |
   = note: forall<T> { Implemented(T: Foo) :- ProjectionEq(<T as std::iter::Iterator>::Item == i32), TypeOutlives(T: 'static), Implemented(T: std::iter::Iterator), Implemented(T: std::marker::Sized). }
```

[rust-lang/rust#69247]: https://github.com/rust-lang/rust/pull/69247
