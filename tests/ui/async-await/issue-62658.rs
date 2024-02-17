// This test created a coroutine whose size was not rounded to a multiple of its
// alignment. This caused an assertion error in codegen.

//@ build-pass
//@ edition:2018

async fn noop() {}

async fn foo() {
    // This suspend should be the largest variant.
    {
        let x = [0u8; 17];
        noop().await;
        println!("{:?}", x);
    }

    // Add one variant that's aligned to 8 bytes.
    {
        let x = 0u64;
        noop().await;
        println!("{:?}", x);
    }
}

fn main() {
    let _ = foo();
}
