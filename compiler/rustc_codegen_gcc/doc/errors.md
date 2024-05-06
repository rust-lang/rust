# Common errors

This file lists errors that were encountered and how to fix them.

### `failed to build archive` error

When you get this error:

```
error: failed to build archive: failed to open object file: No such file or directory (os error 2)
```

That can be caused by the fact that you try to compile with `lto = "fat"`, but you didn't compile the sysroot with LTO.
(Not sure if that's the reason since I cannot reproduce anymore. Maybe it happened when forgetting setting `FAT_LTO`.)

### ld: cannot find crtbegin.o

When compiling an executable with libgccijt, if setting the `*LIBRARY_PATH` variables to the install directory, you will get the following errors:

```
ld: cannot find crtbegin.o: No such file or directory
ld: cannot find -lgcc: No such file or directory
ld: cannot find -lgcc: No such file or directory
libgccjit.so: error: error invoking gcc driver
```

To fix this, set the variables to `gcc-build/build/gcc`.
