This folder contains tests for MIR optimizations.

There are two test formats. One allows specifying a pattern to look for in the MIR, which also
permits leaving placeholders, but requires you to manually change the pattern if anything changes.
The other emits MIR to extra files that you can automatically update by specifying `--bless` on
the command line (just like `ui` tests updating `.stderr` files).

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

# Inline test format

```
(arbitrary rust code)
// END RUST SOURCE
// START $file_name_of_some_mir_dump_0
//  $expected_line_0
// (lines or elision)
// $expected_line_N
// END $file_name_of_some_mir_dump_0
// (lines or elision)
// START $file_name_of_some_mir_dump_N
//  $expected_line_0
// (lines or elision)
// $expected_line_N
// END $file_name_of_some_mir_dump_N
```

All the test information is in comments so the test is runnable.

For each $file_name, compiletest expects [$expected_line_0, ...,
$expected_line_N] to appear in the dumped MIR in order.  Currently it allows
other non-matched lines before and after, but not between $expected_lines,
should you want to skip lines, you must include an elision comment, of the form
(as a regex) `//\s*...\s*`. The lines will be skipped lazily, that is, if there
are two identical lines in the output that match the line after the elision
comment, the first one will be matched.

Examples:

The following blocks will not match the one after it.

```
bb0: {
    StorageLive(_1);
    _1 = const true;
    StorageDead(_1);
}
```

```
bb0: {
    StorageLive(_1);
    _1 = const true;
    goto -> bb1
}
bb1: {
    StorageDead(_1);
    return;
}
```

But this will match the one above,

```
bb0: {
    StorageLive(_1);
    _1 = const true;
    ...
    StorageDead(_1);
    ...
}
```

Lines match ignoring whitespace, and the prefix "//" is removed.

It also currently strips trailing comments -- partly because the full file path
in "scope comments" is unpredictable and partly because tidy complains about
the lines being too long.

compiletest handles dumping the MIR before and after every pass for you.  The
test writer only has to specify the file names of the dumped files (not the
full path to the file) and what lines to expect.  There is an option to rustc
that tells it to dump the mir into some directly (rather then always dumping to
the current directory).
