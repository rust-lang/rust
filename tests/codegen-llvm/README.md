The files here use the LLVM FileCheck framework, documented at
<https://llvm.org/docs/CommandGuide/FileCheck.html>.

If your codegen test has different behavior based on the chosen target or
different compiler flags that you want to exercise, you can set different
filecheck prefixes for each revision:

```rust
// revisions: aaa bbb
// [bbb] compile-flags: --flags-for-bbb
// [aaa] filecheck-flags: --check-prefixes=ALL,AAA
// [bbb] filecheck-flags: --check-prefixes=ALL,BBB
```

After specifying those variations, you can write different expected, or
explicitly *unexpected* output by using `<prefix>-SAME:` and `<prefix>-NOT:`,
like so:

```rust
// ALL: expected code
// AAA-SAME: emitted-only-for-aaa
// AAA-NOT:                        emitted-only-for-bbb
// BBB-NOT:  emitted-only-for-aaa
// BBB-SAME:                       emitted-only-for-bbb
```
