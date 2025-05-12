Attempted to access a private field on a struct.

Erroneous code example:

```compile_fail,E0616
mod some_module {
    pub struct Foo {
        x: u32, // So `x` is private in here.
    }

    impl Foo {
        pub fn new() -> Foo { Foo { x: 0 } }
    }
}

let f = some_module::Foo::new();
println!("{}", f.x); // error: field `x` of struct `some_module::Foo` is private
```

If you want to access this field, you have two options:

1) Set the field public:

```
mod some_module {
    pub struct Foo {
        pub x: u32, // `x` is now public.
    }

    impl Foo {
        pub fn new() -> Foo { Foo { x: 0 } }
    }
}

let f = some_module::Foo::new();
println!("{}", f.x); // ok!
```

2) Add a getter function:

```
mod some_module {
    pub struct Foo {
        x: u32, // So `x` is still private in here.
    }

    impl Foo {
        pub fn new() -> Foo { Foo { x: 0 } }

        // We create the getter function here:
        pub fn get_x(&self) -> &u32 { &self.x }
    }
}

let f = some_module::Foo::new();
println!("{}", f.get_x()); // ok!
```
