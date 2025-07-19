// Repro for <https://github.com/rust-lang/rust/issues/64552#issuecomment-532801232>.
//@ edition: 2021
//@ revisions: assumptions no_assumptions
//@[assumptions] compile-flags: -Zhigher-ranked-assumptions
//@[assumptions] check-pass
//@[no_assumptions] known-bug: #110338

fn needs_send<T: Send>(_val: T) {}
async fn use_async<T>(_val: T) {}

struct MyStruct<'a, T: 'a> {
    val: &'a T,
}

unsafe impl<'a, T: 'a> Send for MyStruct<'a, T> {}

async fn use_my_struct(val: MyStruct<'static, &'static u8>) {
    use_async(val).await;
}

fn main() {
    let first_struct: MyStruct<'static, &'static u8> = MyStruct { val: &&26 };
    let second_struct: MyStruct<'static, &'static u8> = MyStruct { val: &&27 };

    needs_send(first_struct);
    needs_send(use_my_struct(second_struct)); // ERROR
}
