More than one function parameter have the same name.

Erroneous code example:

```compile_fail,E0415
fn foo(f: i32, f: i32) {} // error: identifier `f` is bound more than
                          //        once in this parameter list
```

Please verify you didn't misspell parameters' name. Example:

```
fn foo(f: i32, g: i32) {} // ok!
```
