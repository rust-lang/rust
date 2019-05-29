// ignore-tidy-linelength

#![feature(const_generics)]

#![crate_name = "foo"]

pub enum Order {
    Sorted,
    Unsorted,
}

// @has foo/struct.VSet.html '//pre[@class="rust struct"]' 'pub struct VSet<T, const ORDER: Order>'
// @has foo/struct.VSet.html '//h3[@id="impl-Send"]/code' 'impl<const ORDER: Order, T> Send for VSet<T, ORDER>'
// @has foo/struct.VSet.html '//h3[@id="impl-Sync"]/code' 'impl<const ORDER: Order, T> Sync for VSet<T, ORDER>'
pub struct VSet<T, const ORDER: Order> {
    inner: Vec<T>,
}

// @has foo/struct.VSet.html '//h3[@id="impl"]/code' 'impl<T> VSet<T, { Order::Sorted }>'
impl <T> VSet<T, {Order::Sorted}> {
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }
}

// @has foo/struct.VSet.html '//h3[@id="impl-1"]/code' 'impl<T> VSet<T, { Order::Unsorted }>'
impl <T> VSet<T, {Order::Unsorted}> {
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }
}
