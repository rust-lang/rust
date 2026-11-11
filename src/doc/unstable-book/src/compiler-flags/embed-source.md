# `embed-source`

This flag controls whether the compiler embeds the program source code text into
the object debug information section. It takes one of the following values:

* `y`, `yes`, `on` or `true`: put source code in debug info.
* `n`, `no`, `off`, `false` or no value: omit source code from debug info (the default).

This flag is ignored in configurations that don't emit DWARF debug information
and is ignored on non-LLVM backends. `-Z embed-source` requires DWARFv5. Use
`-Z dwarf-version=5` to control the compiler's DWARF target version and `-g` to
enable debug info generation.
