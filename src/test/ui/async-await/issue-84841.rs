// edition:2018

async fn main() {
    // Adding an .await here avoids the ICE
    test()?;
}

// Removing the const generic parameter here avoids the ICE
async fn test<const N: usize>() {
}
