#![allow(incomplete_features)]
#![feature(adt_const_params)]
#![crate_name = "foo"]

#[derive(PartialEq, Eq)]
pub enum Order {
    Sorted,
    Unsorted,
}

// @has foo/struct.VSet.html '//pre[@class="rust item-decl"]' 'pub struct VSet<T, const ORDER: Order>'
// @has foo/struct.VSet.html '//*[@id="impl-Send-for-VSet%3CT%2C%20ORDER%3E"]/h3[@class="code-header"]' 'impl<T, const ORDER: Order> Send for VSet<T, ORDER>'
// @has foo/struct.VSet.html '//*[@id="impl-Sync-for-VSet%3CT%2C%20ORDER%3E"]/h3[@class="code-header"]' 'impl<T, const ORDER: Order> Sync for VSet<T, ORDER>'
pub struct VSet<T, const ORDER: Order> {
    inner: Vec<T>,
}

// @has foo/struct.VSet.html '//*[@id="impl-VSet%3CT%2C%20{%20Order%3A%3ASorted%20}%3E"]/h3[@class="code-header"]' 'impl<T> VSet<T, { Order::Sorted }>'
impl<T> VSet<T, { Order::Sorted }> {
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }
}

// @has foo/struct.VSet.html '//*[@id="impl-VSet%3CT%2C%20{%20Order%3A%3AUnsorted%20}%3E"]/h3[@class="code-header"]' 'impl<T> VSet<T, { Order::Unsorted }>'
impl<T> VSet<T, { Order::Unsorted }> {
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }
}

pub struct Escape<const S: &'static str>;

// @has foo/struct.Escape.html '//*[@id="impl-Escape%3Cr#%22%3Cscript%3Ealert(%22Escape%22)%3B%3C/script%3E%22#%3E"]/h3[@class="code-header"]' 'impl Escape<r#"<script>alert("Escape");</script>"#>'
impl Escape<r#"<script>alert("Escape");</script>"#> {
    pub fn f() {}
}
