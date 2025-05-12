A struct was declared with two fields having the same name.

Erroneous code example:

```compile_fail,E0124
struct Foo {
    field1: i32,
    field1: i32, // error: field is already declared
}
```

Please verify that the field names have been correctly spelled. Example:

```
struct Foo {
    field1: i32,
    field2: i32, // ok!
}
```
