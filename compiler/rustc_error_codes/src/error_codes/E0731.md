An enum with the representation hint `repr(transparent)` had zero or more than
one variants.

Erroneous code example:

```compile_fail,E0731
#[repr(transparent)]
enum Status { // error: transparent enum needs exactly one variant, but has 2
    Errno(u32),
    Ok,
}
```

Because transparent enums are represented exactly like one of their variants at
run time, said variant must be uniquely determined. If there is no variant, or
if there are multiple variants, it is not clear how the enum should be
represented.
