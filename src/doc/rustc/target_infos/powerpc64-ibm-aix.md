---
maintainers: [
    "QIU Chaofan `qiucofan@cn.ibm.com`, https://github.com/ecnelises",
    "Kai LUO, `lkail@cn.ibm.com`, https://github.com/bzEq",
]
---

## Requirements

This target supports host tools, std and alloc. This target cannot be cross-compiled as for now, mainly because of the unavailability of system linker on other platforms.

Binary built for this target is expected to run on Power7 or newer CPU, and AIX 7.2 or newer version.

Binary format of this platform is XCOFF. Archive file format is 'AIX big format'.

## Testing

This target supports running test suites natively, but it's not available to cross-compile and execute in emulator.
