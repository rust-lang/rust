//@ check-pass

// Extending `improper_ctypes` to check external-ABI fn-ptrs means that it can encounter
// projections which cannot be normalized - unsurprisingly, this shouldn't crash the compiler.

trait Bar {
    type Assoc;
}

type Foo<T> = extern "C" fn() -> <T as Bar>::Assoc;

fn main() {}
