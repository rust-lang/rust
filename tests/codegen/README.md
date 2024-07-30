The files here use the LLVM FileCheck framework, documented at
<https://llvm.org/docs/CommandGuide/FileCheck.html>.

In short, directives are added to the code file that will be used to validate
the code output. Common directives include:

- `CHECK-LABEL s`: used to identify logical blocks, e.g. function names.
- `CHECK s`: verify that a line matching `s` exists.
- `CHECK-SAME s`: verify that the line matched by the previous directive
  matches`s`.
- `CHECK-NEXT s`: verify that the next line immediately after the previous match
  matches `s`.
- `CHECK-NOT s`: verify that `s` is _not_ found.

One extension worth noting is the use of revisions as custom prefixes for
FileCheck. If your codegen test has different behavior based on the chosen
target or different compiler flags that you want to exercise, you can use a
revisions annotation, like so:

```rust
//@ revisions: aaa bbb
//@ [aaa] compile-flags: --target some-target-kebab --other-flags
//@ [bbb] compile-flags: --target other-target-kebab --flags-for-bbb
```

These directives will be usable by replacing `CHECK` with
`CHECK-{revision.to_uppercase()}`:

- `CHECK-AAA s`: verify that `s` is found, only when running revision `aaa`
- `CHECK-AAA-NEXT s`: verify that `s` is found after the preceding match, only
  when running revision `aaa`
- ...

For example, if you wanted to validate the following with `aaa`'s flags:'

```text
my func:
    #
    # lines you don't care about
    #

    [found on both] [aaa only]
    aaa finished
```

And this with `bbb`'s:'

```text
my_func:
    [found on both] [bbb only]
    bbb finished
```

The following should work:

```rust
// CHECK-LABEL: my_func:
// CHECK: [found on both]
// CHECK-AAA-SAME: [aaa only]
// CHECK-AAA-NOT:  [bbb only]
// CHECK-BBB-SAME: [bbb only]
// CHECK-BBB-NOT:  [aaa only]
// CHECK-AAA-NEXT: aaa finished
// CHECK-BBB-NEXT: bbbfinished
```
