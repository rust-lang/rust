This folder contains tests for MIR optimizations.

The test format is:

```
(arbitrary rust code)
// END RUST SOURCE
// START $file_name_of_some_mir_dump_0
//  $expected_line_0
// ...
// $expected_line_N
// END $file_name_of_some_mir_dump_0
// ...
// START $file_name_of_some_mir_dump_N
//  $expected_line_0
// ...
// $expected_line_N
// END $file_name_of_some_mir_dump_N
```

All the test information is in comments so the test is runnable.

For each $file_name, compiletest expects [$expected_line_0, ...,
$expected_line_N] to appear in the dumped MIR in order.  Currently it allows
other non-matched lines before, after and in-between. Note that this includes
lines that end basic blocks or begin new ones; it is good practice
in your tests to include the terminator for each of your basic blocks as an
internal sanity check guarding against a test like:

```
bb0: {
    StorageLive(_1);
    _1 = const true;
    StorageDead(_1);
}
```

that will inadvertantly pattern-matching against:

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

Lines match ignoring whitespace, and the prefix "//" is removed.

It also currently strips trailing comments -- partly because the full file path
in "scope comments" is unpredictable and partly because tidy complains about
the lines being too long.

compiletest handles dumping the MIR before and after every pass for you.  The
test writer only has to specify the file names of the dumped files (not the
full path to the file) and what lines to expect.  I added an option to rustc
that tells it to dump the mir into some directly (rather then always dumping to
the current directory).  

Lines match ignoring whitespace, and the prefix "//" is removed of course.

It also currently strips trailing comments -- partly because the full file path
in "scope comments" is unpredictable and partly because tidy complains about
the lines being too long.

