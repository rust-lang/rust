A pattern was declared as an argument in a foreign function declaration.

Erroneous code example:

```compile_fail,E0130
extern "C" {
    fn foo((a, b): (u32, u32)); // error: patterns aren't allowed in foreign
                                //        function declarations
}
```

To fix this error, replace the pattern argument with a regular one. Example:

```
struct SomeStruct {
    a: u32,
    b: u32,
}

extern "C" {
    fn foo(s: SomeStruct); // ok!
}
```

Or:

```
extern "C" {
    fn foo(a: (u32, u32)); // ok!
}
```
