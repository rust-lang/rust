A non-mutable value was assigned a value.

Erroneous code example:

```compile_fail,E0594
struct SolarSystem {
    earth: i32,
}

let ss = SolarSystem { earth: 3 };
ss.earth = 2; // error!
```

To fix this error, declare `ss` as mutable by using the `mut` keyword:

```
struct SolarSystem {
    earth: i32,
}

let mut ss = SolarSystem { earth: 3 }; // declaring `ss` as mutable
ss.earth = 2; // ok!
```
