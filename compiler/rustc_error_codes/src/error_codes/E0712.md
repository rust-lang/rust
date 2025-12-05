A borrow of a thread-local variable was made inside a function which outlived
the lifetime of the function.

Erroneous code example:

```compile_fail,E0712
#![feature(thread_local)]

#[thread_local]
static FOO: u8 = 3;

fn main() {
    let a = &FOO; // error: thread-local variable borrowed past end of function

    std::thread::spawn(move || {
        println!("{}", a);
    });
}
```
