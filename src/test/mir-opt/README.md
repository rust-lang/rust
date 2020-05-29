This folder contains tests for MIR optimizations.

The `mir-opt` test format emits MIR to extra files that you can automatically update by specifying
`--bless` on the command line (just like `ui` tests updating `.stderr` files).

# `--bless`able test format

By default 32 bit and 64 bit targets use the same dump files, which can be problematic in the
presence of pointers in constants or other bit width dependent things. In that case you can add

```
// EMIT_MIR_FOR_EACH_BIT_WIDTH
```

to your test, causing separate files to be generated for 32bit and 64bit systems.

## Emit a diff of the mir for a specific optimization

This is what you want most often when you want to see how an optimization changes the MIR.

```
// EMIT_MIR $file_name_of_some_mir_dump.diff
```

## Emit mir after a specific optimization

Use this if you are just interested in the final state after an optimization.

```
// EMIT_MIR $file_name_of_some_mir_dump.after.mir
```

## Emit mir before a specific optimization

This exists mainly for completeness and is rarely useful.

```
// EMIT_MIR $file_name_of_some_mir_dump.before.mir
```
