//@ edition: 2021
//@ check-pass

// Make sure we don't cycle error when normalizing types for tail expr drop order lint.

#![deny(tail_expr_drop_order)]

async fn test() -> Result<(), Box<dyn std::error::Error>> {
    Box::pin(test()).await?;
    Ok(())
}

fn main() {}
