# Supported `RUSTFLAGS`

To support you while debugging or profiling, we have added support for an experimental `-Z autodiff` rustc flag (which can be passed to cargo via `RUSTFLAGS`), which allow changing the behaviour of Enzyme, without recompiling rustc. We currently support the following values for `autodiff`.

### Debug Flags

```text
PrintTA // Print TypeAnalysis information
PrintAA // Print ActivityAnalysis information
Print // Print differentiated functions while they are being generated and optimized
PrintPerf // Print AD related Performance warnings
PrintModBefore // Print the whole LLVM-IR module directly before running AD
PrintModAfter // Print the whole LLVM-IR module after running AD, before optimizations
PrintModFinal // Print the whole LLVM-IR module after running optimizations and AD
LooseTypes // Risk incorrect derivatives instead of aborting when missing Type Info 
```

<div class="warning">
`LooseTypes` is often helpful to get rid of Enzyme errors stating `Can not deduce type of <X>` and to be able to run some code. But please keep in mind that this flag absolutely has the chance to cause incorrect gradients. Even worse, the gradients might be correct for certain input values, but not for others. So please create issues about such bugs and only use this flag temporarily while you wait for your bug to be fixed.
</div>

### Benchmark flags

For performance experiments and benchmarking we also support

```text
NoPostopt // We won't optimize the LLVM-IR Module after AD
RuntimeActivity // Enables the runtime activity feature from Enzyme 
Inline // Instructs Enzyme to maximize inlining as far as possible, beyond LLVM's default
```

You can combine multiple `autodiff` values using a comma as separator:

```bash
RUSTFLAGS="-Z autodiff=Enable,LooseTypes,PrintPerf" cargo +enzyme build
```

Using `-Zautodiff=Enable` will allow using autodiff and update your normal rustc compilation pipeline:

1. Run your selected compilation pipeline. If you selected a release build, we will disable vectorization and loop unrolling.
2. Differentiate your functions.
3. Run your selected compilation pipeline again on the whole module. This time we do not disable vectorization or loop unrolling.
