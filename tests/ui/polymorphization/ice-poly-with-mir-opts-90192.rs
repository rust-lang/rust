// issue: rust-lang/rust#90192
// ICE assertion failed: matches!(ty.kind(), ty :: Param(_))
//@ compile-flags:-Zpolymorphize=on -Zmir-opt-level=3
//@ build-pass

fn test<T>() {
    std::mem::size_of::<T>();
}

pub fn foo<T>(_: T) -> &'static fn() {
    &(test::<T> as fn())
}

fn outer<T>() {
    foo(|| ());
}

fn main() {
    outer::<u8>();
}
