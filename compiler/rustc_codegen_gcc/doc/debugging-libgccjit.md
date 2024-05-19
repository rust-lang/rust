# Debugging libgccjit

Sometimes, libgccjit will crash and output an error like this:

```
during RTL pass: expand
libgccjit.so: error: in expmed_mode_index, at expmed.h:249
0x7f0da2e61a35 expmed_mode_index
	../../../gcc/gcc/expmed.h:249
0x7f0da2e61aa4 expmed_op_cost_ptr
	../../../gcc/gcc/expmed.h:271
0x7f0da2e620dc sdiv_cost_ptr
	../../../gcc/gcc/expmed.h:540
0x7f0da2e62129 sdiv_cost
	../../../gcc/gcc/expmed.h:558
0x7f0da2e73c12 expand_divmod(int, tree_code, machine_mode, rtx_def*, rtx_def*, rtx_def*, int)
	../../../gcc/gcc/expmed.c:4335
0x7f0da2ea1423 expand_expr_real_2(separate_ops*, rtx_def*, machine_mode, expand_modifier)
	../../../gcc/gcc/expr.c:9240
0x7f0da2cd1a1e expand_gimple_stmt_1
	../../../gcc/gcc/cfgexpand.c:3796
0x7f0da2cd1c30 expand_gimple_stmt
	../../../gcc/gcc/cfgexpand.c:3857
0x7f0da2cd90a9 expand_gimple_basic_block
	../../../gcc/gcc/cfgexpand.c:5898
0x7f0da2cdade8 execute
	../../../gcc/gcc/cfgexpand.c:6582
```

To see the code which causes this error, call the following function:

```c
gcc_jit_context_dump_to_file(ctxt, "/tmp/output.c", 1 /* update_locations */)
```

This will create a C-like file and add the locations into the IR pointing to this C file.
Then, rerun the program and it will output the location in the second line:

```
libgccjit.so: /tmp/something.c:61322:0: error: in expmed_mode_index, at expmed.h:249
```

Or add a breakpoint to `add_error` in gdb and print the line number using:

```
p loc->m_line
p loc->m_filename->m_buffer
```

To print a debug representation of a tree:

```c
debug_tree(expr);
```

(defined in print-tree.h)

To print a debug representation of a gimple struct:

```c
debug_gimple_stmt(gimple_struct)
```

To get the `rustc` command to run in `gdb`, add the `--verbose` flag to `cargo build`.

To have the correct file paths in `gdb` instead of `/usr/src/debug/gcc/libstdc++-v3/libsupc++/eh_personality.cc`:

Maybe by calling the following at the beginning of gdb:

```
set substitute-path /usr/src/debug/gcc /path/to/gcc-repo/gcc
```

TODO(antoyo): but that's not what I remember I was doing.
