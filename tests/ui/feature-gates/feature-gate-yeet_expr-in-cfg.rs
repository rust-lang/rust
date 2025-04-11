//@ edition: 2021

pub fn demo() -> Option<i32> {
    #[cfg(false)]
    {
        do yeet //~ ERROR `do yeet` expression is experimental
    }

    Some(1)
}

#[cfg(false)]
pub fn alternative() -> Result<(), String> {
    do yeet "hello"; //~ ERROR `do yeet` expression is experimental
}

fn main() {
    demo();
}
