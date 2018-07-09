# MIR passes

If you would like to get the MIR for a function (or constant, etc),
you can use the `optimized_mir(def_id)` query. This will give you back
the final, optimized MIR. For foreign def-ids, we simply read the MIR
from the other crate's metadata. But for local def-ids, the query will
construct the MIR and then iteratively optimize it by applying a
series of passes. This section describes how those passes work and how
you can extend them.

To produce the `optimized_mir(D)` for a given def-id `D`, the MIR
passes through several suites of optimizations, each represented by a
query. Each suite consists of multiple optimizations and
transformations. These suites represent useful intermediate points
where we want to access the MIR for type checking or other purposes:

- `mir_build(D)` – not a query, but this constructs the initial MIR
- `mir_const(D)` – applies some simple transformations to make MIR ready for
  constant evaluation;
- `mir_validated(D)` – applies some more transformations, making MIR ready for
  borrow checking;
- `optimized_mir(D)` – the final state, after all optimizations have been
  performed.

### Seeing how the MIR changes as the compiler executes

`-Zdump-mir=F` is a handy compiler options that will let you view the MIR for
each function at each stage of compilation. `-Zdump-mir` takes a **filter** `F`
which allows you to control which functions and which passes you are
interesting in. For example:

```bash
> rustc -Zdump-mir=foo ...
```

This will dump the MIR for any function whose name contains `foo`; it
will dump the MIR both before and after every pass. Those files will
be created in the `mir_dump` directory. There will likely be quite a
lot of them!

```bash
> cat > foo.rs
fn main() {
    println!("Hello, world!");
}
^D
> rustc -Zdump-mir=main foo.rs
> ls mir_dump/* | wc -l
     161
```

The files have names like `rustc.main.000-000.CleanEndRegions.after.mir`. These
names have a number of parts:

```text
rustc.main.000-000.CleanEndRegions.after.mir
      ---- --- --- --------------- ----- either before or after
      |    |   |   name of the pass
      |    |   index of dump within the pass (usually 0, but some passes dump intermediate states)
      |    index of the pass
      def-path to the function etc being dumped
```

You can also make more selective filters. For example, `main & CleanEndRegions`
will select for things that reference *both* `main` and the pass
`CleanEndRegions`:

```bash
> rustc -Zdump-mir='main & CleanEndRegions' foo.rs
> ls mir_dump
rustc.main.000-000.CleanEndRegions.after.mir	rustc.main.000-000.CleanEndRegions.before.mir
```

Filters can also have `|` parts to combine multiple sets of
`&`-filters. For example `main & CleanEndRegions | main &
NoLandingPads` will select *either* `main` and `CleanEndRegions` *or*
`main` and `NoLandingPads`:

```bash
> rustc -Zdump-mir='main & CleanEndRegions | main & NoLandingPads' foo.rs
> ls mir_dump
rustc.main-promoted[0].002-000.NoLandingPads.after.mir
rustc.main-promoted[0].002-000.NoLandingPads.before.mir
rustc.main-promoted[0].002-006.NoLandingPads.after.mir
rustc.main-promoted[0].002-006.NoLandingPads.before.mir
rustc.main-promoted[1].002-000.NoLandingPads.after.mir
rustc.main-promoted[1].002-000.NoLandingPads.before.mir
rustc.main-promoted[1].002-006.NoLandingPads.after.mir
rustc.main-promoted[1].002-006.NoLandingPads.before.mir
rustc.main.000-000.CleanEndRegions.after.mir
rustc.main.000-000.CleanEndRegions.before.mir
rustc.main.002-000.NoLandingPads.after.mir
rustc.main.002-000.NoLandingPads.before.mir
rustc.main.002-006.NoLandingPads.after.mir
rustc.main.002-006.NoLandingPads.before.mir
```

(Here, the `main-promoted[0]` files refer to the MIR for "promoted constants"
that appeared within the `main` function.)

### Implementing and registering a pass

A `MirPass` is some bit of code that processes the MIR, typically –
but not always – transforming it along the way somehow. For example,
it might perform an optimization. The `MirPass` trait itself is found
in in [the `rustc_mir::transform` module][mirtransform], and it
basically consists of one method, `run_pass`, that simply gets an
`&mut Mir` (along with the tcx and some information about where it
came from). The MIR is therefore modified in place (which helps to
keep things efficient).

A good example of a basic MIR pass is [`NoLandingPads`], which walks
the MIR and removes all edges that are due to unwinding – this is
used when configured with `panic=abort`, which never unwinds. As you
can see from its source, a MIR pass is defined by first defining a
dummy type, a struct with no fields, something like:

```rust
struct MyPass;
```

for which you then implement the `MirPass` trait. You can then insert
this pass into the appropriate list of passes found in a query like
`optimized_mir`, `mir_validated`, etc. (If this is an optimization, it
should go into the `optimized_mir` list.)

If you are writing a pass, there's a good chance that you are going to
want to use a [MIR visitor]. MIR visitors are a handy way to walk all
the parts of the MIR, either to search for something or to make small
edits.

### Stealing

The intermediate queries `mir_const()` and `mir_validated()` yield up
a `&'tcx Steal<Mir<'tcx>>`, allocated using
`tcx.alloc_steal_mir()`. This indicates that the result may be
**stolen** by the next suite of optimizations – this is an
optimization to avoid cloning the MIR. Attempting to use a stolen
result will cause a panic in the compiler. Therefore, it is important
that you do not read directly from these intermediate queries except as
part of the MIR processing pipeline.

Because of this stealing mechanism, some care must also be taken to
ensure that, before the MIR at a particular phase in the processing
pipeline is stolen, anyone who may want to read from it has already
done so. Concretely, this means that if you have some query `foo(D)`
that wants to access the result of `mir_const(D)` or
`mir_validated(D)`, you need to have the successor pass "force"
`foo(D)` using `ty::queries::foo::force(...)`. This will force a query
to execute even though you don't directly require its result.

As an example, consider MIR const qualification. It wants to read the
result produced by the `mir_const()` suite. However, that result will
be **stolen** by the `mir_validated()` suite. If nothing was done,
then `mir_const_qualif(D)` would succeed if it came before
`mir_validated(D)`, but fail otherwise. Therefore, `mir_validated(D)`
will **force** `mir_const_qualif` before it actually steals, thus
ensuring that the reads have already happened (remember that
[queries are memoized](./query.html), so executing a query twice
simply loads from a cache the second time):

```text
mir_const(D) --read-by--> mir_const_qualif(D)
     |                       ^
  stolen-by                  |
     |                    (forces)
     v                       |
mir_validated(D) ------------+
```

This mechanism is a bit dodgy. There is a discussion of more elegant
alternatives in [rust-lang/rust#41710].

[rust-lang/rust#41710]: https://github.com/rust-lang/rust/issues/41710
[mirtransform]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/transform/
[`NoLandingPads`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/transform/no_landing_pads/struct.NoLandingPads.html
[MIR visitor]: mir/visitor.html
