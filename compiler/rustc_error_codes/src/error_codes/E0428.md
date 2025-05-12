A type or module has been defined more than once.

Erroneous code example:

```compile_fail,E0428
struct Bar;
struct Bar; // error: duplicate definition of value `Bar`
```

Please verify you didn't misspell the type/module's name or remove/rename the
duplicated one. Example:

```
struct Bar;
struct Bar2; // ok!
```
