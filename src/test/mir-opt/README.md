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
other non-matched lines before, after and in-between.  

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

