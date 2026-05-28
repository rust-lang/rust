//@ build-pass
//@ edition:2018

#![feature(coroutines)]

fn main() {
    foo();
}

fn foo() {
    #[coroutine] || {
        yield drop(Config {
            nickname: NonCopy,
            b: NonCopy2,
        }.nickname);
    };
}

#[derive(Default)]
struct NonCopy;
impl Drop for NonCopy {
    fn drop(&mut self) {}
}

#[derive(Default)]
struct NonCopy2;
impl Drop for NonCopy2 {
    fn drop(&mut self) {}
}

#[derive(Default)]
struct Config {
    nickname: NonCopy,
    b: NonCopy2,
}
