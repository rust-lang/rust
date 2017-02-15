# The `Deref` trait

The `Deref<Target = U>` trait allows a type to implicitly implement all the methods
of the type `U`. When attempting to resolve a method call, the compiler will search
the top-level type for the implementation of the called method. If no such method is
found, `.deref()` is called and the compiler continues to search for the method
implementation in the returned type `U`.
