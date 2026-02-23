# (WIP) LLVM Codegen

When Rust calls an LLVM `DIBuilder` function, LLVM translates the given information to a
["debug record"][dbg_record] that is format-agnostic. These records can be inspected in the LLVM-IR.

[dbg_record]: https://llvm.org/docs/SourceLevelDebugging.html#debug-records

It is important to note that tags within the debug records are **always stored as DWARF tags**. If
the target calls for PDB debug info, during codegen the debug records will then be passed through
[a module that translates the DWARF tags to their CodeView counterparts][cv].

[cv]:https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/AsmPrinter/CodeViewDebug.cpp