//@ edition: 2018

pub fn demo() -> Option<i32> {
    do yeet //~ ERROR `do yeet` expression is experimental
}

pub fn main() -> Result<(), String> {
    do yeet "hello"; //~ ERROR `do yeet` expression is experimental
}
