# `embed-source`

This flag controls whether the compiler embeds the program source code text into
object debug info section. It takes one of the following values:

* `y`, `yes`, `on` or `true`: put source code in debug info.
* `n`, `no`, `off`, `false` or no value: omit source code from debug info (the default).

`-C embed-source` requires DWARFv5 and is only supported with the LLVM backend.
Use `-Z dwarf-version=5` to control the compiler's DWARF target version.