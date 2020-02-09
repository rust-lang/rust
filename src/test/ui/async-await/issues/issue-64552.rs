// edition: 2018

async fn use_async<T>(_val: T) {}

struct MyStruct<'a, T: 'a> {
    val: &'a T
}

unsafe impl<'a, T: 'a> Send for MyStruct<'a, T> {}

async fn use_my_struct(val: MyStruct<'static, &'static u8>) {
    use_async(val).await;
}

fn main() {
    let first_struct: MyStruct<'static, &'static u8> = MyStruct { val: &&26 };
    needs_send(use_my_struct(second_struct));
}
