The files here use the LLVM FileCheck framework, documented at
<https://llvm.org/docs/CommandGuide/FileCheck.html>.

One extension worth noting is the use of revisions as custom prefixes for
FileCheck. If your codegen test has different behavior based on the chosen
target or different compiler flags that you want to exercise, you can use a
revisions annotation, like so:

```rust
// revisions: aaa bbb
// [bbb] compile-flags: --flags-for-bbb
```

After specifying those variations, you can write different expected, or
explicitly *unexpected* output by using `<prefix>-SAME:` and `<prefix>-NOT:`,
like so:

```rust
// CHECK: expected code
// aaa-SAME: emitted-only-for-aaa
// aaa-NOT:                        emitted-only-for-bbb
// bbb-NOT:  emitted-only-for-aaa
// bbb-SAME:                       emitted-only-for-bbb
```
