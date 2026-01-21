//@ edition: 2024

#![crate_name = "async_dep"]
#![crate_type = "rlib"]

#[cfg(rpass1)]
async fn private_helper() -> u32 {
    42
}

#[cfg(rpass2)]
async fn private_helper() -> u32 {
    let x = 21;
    let y = 21;
    x + y
}

#[cfg(rpass3)]
async fn private_helper() -> u32 {
    async { 42 }.await
}

pub async fn public_async_fn() -> u32 {
    private_helper().await
}

pub struct AsyncStruct;

impl AsyncStruct {
    pub async fn method(&self) -> u32 {
        private_helper().await
    }
}
