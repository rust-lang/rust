An unknown field was specified into a structure.

Erroneous code example:

```compile_fail,E0560
struct Simba {
    mother: u32,
}

let s = Simba { mother: 1, father: 0 };
// error: structure `Simba` has no field named `father`
```

Verify you didn't misspell the field's name or that the field exists. Example:

```
struct Simba {
    mother: u32,
    father: u32,
}

let s = Simba { mother: 1, father: 0 }; // ok!
```
