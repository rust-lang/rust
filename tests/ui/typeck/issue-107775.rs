//@ edition: 2021

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

pub trait Trait {
    fn do_something<'async_trait>(byte: u8)
    ->
        Pin<Box<dyn Future<Output = ()> +
        Send + 'async_trait>>;
}

pub struct Struct;

impl Trait for Struct {
    fn do_something<'async_trait>(byte: u8)
        ->
            Pin<Box<dyn Future<Output = ()> +
            Send + 'async_trait>> {
        Box::pin(

            async move { let byte = byte; let _: () = {}; })
    }
}

pub struct Map {
    map: HashMap<u16, fn(u8) -> Pin<Box<dyn Future<Output = ()> + Send>>>,
}

impl Map {
    pub fn new() -> Self {
        let mut map = HashMap::new();
        map.insert(1, Struct::do_something);
        Self { map }
        //~^ ERROR mismatched types
    }
}

fn main() {}
