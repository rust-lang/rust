Generic arguments were not provided in the same order as the corresponding
generic parameters are declared.

Erroneous code example:

```compile_fail,E0747
struct S<'a, T>(&'a T);

type X = S<(), 'static>; // error: the type argument is provided before the
                         // lifetime argument
```

The argument order should be changed to match the parameter declaration
order, as in the following:

```
struct S<'a, T>(&'a T);

type X = S<'static, ()>; // ok
```
