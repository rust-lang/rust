A private item was used outside of its scope.

Erroneous code example:

```compile_fail,E0624
mod inner {
    pub struct Foo;

    impl Foo {
        fn method(&self) {}
    }
}

let foo = inner::Foo;
foo.method(); // error: method `method` is private
```

Two possibilities are available to solve this issue:

1. Only use the item in the scope it has been defined:

```
mod inner {
    pub struct Foo;

    impl Foo {
        fn method(&self) {}
    }

    pub fn call_method(foo: &Foo) { // We create a public function.
        foo.method(); // Which calls the item.
    }
}

let foo = inner::Foo;
inner::call_method(&foo); // And since the function is public, we can call the
                          // method through it.
```

2. Make the item public:

```
mod inner {
    pub struct Foo;

    impl Foo {
        pub fn method(&self) {} // It's now public.
    }
}

let foo = inner::Foo;
foo.method(); // Ok!
```
