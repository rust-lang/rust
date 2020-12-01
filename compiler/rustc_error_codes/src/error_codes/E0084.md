An unsupported representation was attempted on a zero-variant enum.

Erroneous code example:

```compile_fail,E0084
#[repr(i32)]
enum NightsWatch {} // error: unsupported representation for zero-variant enum
```

It is impossible to define an integer type to be used to represent zero-variant
enum values because there are no zero-variant enum values. There is no way to
construct an instance of the following type using only safe code. So you have
two solutions. Either you add variants in your enum:

```
#[repr(i32)]
enum NightsWatch {
    JonSnow,
    Commander,
}
```

or you remove the integer representation of your enum:

```
enum NightsWatch {}
```
