`align` representation hint was incorrectly declared.

Erroneous code examples:

```compile_fail,E0693
#[repr(align=8)] // error!
struct Align8(i8);

#[repr(align="8")] // error!
struct Align8(i8);
```

This is a syntax error at the level of attribute declarations. The proper
syntax for `align` representation hint is the following:

```
#[repr(align(8))] // ok!
struct Align8(i8);
```
