% Tuple Structs

Rust has another data type that's like a hybrid between a tuple and a struct,
called a *tuple struct*. Tuple structs do have a name, but their fields don't:

```{rust}
struct Color(i32, i32, i32);
struct Point(i32, i32, i32);
```

These two will not be equal, even if they have the same values:

```{rust}
# struct Color(i32, i32, i32);
# struct Point(i32, i32, i32);
let black = Color(0, 0, 0);
let origin = Point(0, 0, 0);
```

It is almost always better to use a struct than a tuple struct. We would write
`Color` and `Point` like this instead:

```{rust}
struct Color {
    red: i32,
    blue: i32,
    green: i32,
}

struct Point {
    x: i32,
    y: i32,
    z: i32,
}
```

Now, we have actual names, rather than positions. Good names are important,
and with a struct, we have actual names.

There _is_ one case when a tuple struct is very useful, though, and that's a
tuple struct with only one element. We call this the *newtype* pattern, because
it allows you to create a new type, distinct from that of its contained value
and expressing its own semantic meaning:

```{rust}
struct Inches(i32);

let length = Inches(10);

let Inches(integer_length) = length;
println!("length is {} inches", integer_length);
```

As you can see here, you can extract the inner integer type through a
destructuring `let`, as we discussed previously in 'tuples.' In this case, the
`let Inches(integer_length)` assigns `10` to `integer_length`.
