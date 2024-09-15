//@ known-bug: rust-lang/rust#126939

struct MySlice<T>(T);
type MySliceBool = MySlice<[bool]>;

struct P2 {
    b: MySliceBool,
}

static CHECK: () = assert!(align_of::<P2>() == 1);

fn main() {}
