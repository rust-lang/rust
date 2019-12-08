A struct constructor with private fields was invoked.

Erroneous code example:

```compile_fail,E0451
mod Bar {
    pub struct Foo {
        pub a: isize,
        b: isize,
    }
}

let f = Bar::Foo{ a: 0, b: 0 }; // error: field `b` of struct `Bar::Foo`
                                //        is private
```

To fix this error, please ensure that all the fields of the struct are public,
or implement a function for easy instantiation. Examples:

```
mod Bar {
    pub struct Foo {
        pub a: isize,
        pub b: isize, // we set `b` field public
    }
}

let f = Bar::Foo{ a: 0, b: 0 }; // ok!
```

Or:

```
mod Bar {
    pub struct Foo {
        pub a: isize,
        b: isize, // still private
    }

    impl Foo {
        pub fn new() -> Foo { // we create a method to instantiate `Foo`
            Foo { a: 0, b: 0 }
        }
    }
}

let f = Bar::Foo::new(); // ok!
```
