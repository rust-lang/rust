This folder contains tests for MIR optimizations.

The test format is:

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
