A private item was used outside its scope.

Erroneous code example:

```compile_fail,E0603
mod SomeModule {
    const PRIVATE: u32 = 0x_a_bad_1dea_u32; // This const is private, so we
                                            // can't use it outside of the
                                            // `SomeModule` module.
}

println!("const value: {}", SomeModule::PRIVATE); // error: constant `PRIVATE`
                                                  //        is private
```

In order to fix this error, you need to make the item public by using the `pub`
keyword. Example:

```
mod SomeModule {
    pub const PRIVATE: u32 = 0x_a_bad_1dea_u32; // We set it public by using the
                                                // `pub` keyword.
}

println!("const value: {}", SomeModule::PRIVATE); // ok!
```
