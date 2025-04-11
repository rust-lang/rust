//@ edition: 2021

pub fn demo() -> Option<i32> {
    #[cfg(FALSE)]
    {
        do yeet //~ ERROR `do yeet` expression is experimental
    }

    Some(1)
}

#[cfg(FALSE)]
pub fn alternative() -> Result<(), String> {
    do yeet "hello"; //~ ERROR `do yeet` expression is experimental
}

fn main() {
    demo();
}
