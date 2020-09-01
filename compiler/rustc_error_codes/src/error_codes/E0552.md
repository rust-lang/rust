A unrecognized representation attribute was used.

Erroneous code example:

```compile_fail,E0552
#[repr(D)] // error: unrecognized representation hint
struct MyStruct {
    my_field: usize
}
```

You can use a `repr` attribute to tell the compiler how you want a struct or
enum to be laid out in memory.

Make sure you're using one of the supported options:

```
#[repr(C)] // ok!
struct MyStruct {
    my_field: usize
}
```

For more information about specifying representations, see the ["Alternative
Representations" section] of the Rustonomicon.

["Alternative Representations" section]: https://doc.rust-lang.org/nomicon/other-reprs.html
