This error occurs because a value was dropped while it was still borrowed.

Erroneous code example:

```compile_fail,E0597
struct Foo<'a> {
    x: Option<&'a u32>,
}

let mut x = Foo { x: None };
{
    let y = 0;
    x.x = Some(&y); // error: `y` does not live long enough
}
println!("{:?}", x.x);
```

Here, `y` is dropped at the end of the inner scope, but it is borrowed by
`x` until the `println`. To fix the previous example, just remove the scope
so that `y` isn't dropped until after the println

```
struct Foo<'a> {
    x: Option<&'a u32>,
}

let mut x = Foo { x: None };

let y = 0;
x.x = Some(&y);

println!("{:?}", x.x);
```
