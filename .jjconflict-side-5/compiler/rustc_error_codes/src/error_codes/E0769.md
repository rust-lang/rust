A tuple struct or tuple variant was used in a pattern as if it were a struct or
struct variant.

Erroneous code example:

```compile_fail,E0769
enum E {
    A(i32),
}

let e = E::A(42);

match e {
    E::A { number } => { // error!
        println!("{}", number);
    }
}
```

To fix this error, you can use the tuple pattern:

```
# enum E {
#     A(i32),
# }
# let e = E::A(42);
match e {
    E::A(number) => { // ok!
        println!("{}", number);
    }
}
```

Alternatively, you can also use the struct pattern by using the correct field
names and binding them to new identifiers:

```
# enum E {
#     A(i32),
# }
# let e = E::A(42);
match e {
    E::A { 0: number } => { // ok!
        println!("{}", number);
    }
}
```
