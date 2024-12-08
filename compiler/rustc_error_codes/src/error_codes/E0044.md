You cannot use type or const parameters on foreign items.

Example of erroneous code:

```compile_fail,E0044
extern "C" { fn some_func<T>(x: T); }
```

To fix this, replace the generic parameter with the specializations that you
need:

```
extern "C" { fn some_func_i32(x: i32); }
extern "C" { fn some_func_i64(x: i64); }
```
