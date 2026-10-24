An unknown field was specified into an enum's structure variant.

Erroneous code example:

```compile_fail,E0559
enum Field {
    Fool { x: u32 },
}

let s = Field::Fool { joke: 0 };
// error: struct variant `Field::Fool` has no field named `joke`
```

Verify you didn't misspell the field's name or that the field exists. Example:

```
enum Field {
    Fool { joke: u32 },
}

let s = Field::Fool { joke: 0 }; // ok!
```
