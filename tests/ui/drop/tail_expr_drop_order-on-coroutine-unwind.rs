//@ edition: 2021

// Make sure we don't ICE when emitting the "lint" drop statement
// used for tail_expr_drop_order.

#![deny(tail_expr_drop_order)]

struct Drop;
impl std::ops::Drop for Drop {
    fn drop(&mut self) {}
}

async fn func() -> Result<(), Drop> {
    todo!()
}

async fn retry_db() -> Result<(), Drop> {
    loop {
        match func().await {
            //~^ ERROR relative drop order changing in Rust 2024
            //~| WARNING this changes meaning in Rust 2024
            Ok(()) => return Ok(()),
            Err(e) => {}
        }
    }
}

fn main() {}
