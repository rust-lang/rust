fn main(args: [str]) {
    let one = fn@() -> uint {
        enum r { a };
        ret a as uint;
    };
    let two = fn@() -> uint {
        enum r { a };
        ret a as uint;
    };
    one(); two();
}
