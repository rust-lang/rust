# Line number information

Line number information maps between machine code instructions and the source level location.

## Encoding

The line number information is stored in the `.debug_line` section for ELF and `__debug_line`
section of the `__DWARF` segment for Mach-O object files. The line number information contains a
header followed by the line program. The line program is a program for a virtual machine with
instructions like set line number for the current machine code instruction and advance the current
machine code instruction.

## Tips

You need to set either `DW_AT_low_pc` and `DW_AT_high_pc` **or** `DW_AT_ranges` of a
`DW_TAG_compilation_unit` to the range of addresses in the compilation unit. After that you need
to set `DW_AT_stmt_list` to the `.debug_line` section offset of the line program. Otherwise a
debugger won't find the line number information. On macOS the debuginfo relocations **must** be
section relative and not symbol relative.
See [#303 (comment)](https://github.com/bjorn3/rustc_codegen_cranelift/issues/303#issuecomment-457825535)
for more information.
