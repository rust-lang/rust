// compile-flags: -Z parse-only

#[path() token] //~ ERROR expected `]`, found `token`
mod m {}
