mod foo {
    pub const X: u32 = 1;
}

pub use foo as foo2;
//~^ ERROR `foo` is private, and cannot be re-exported [E0365]

fn main() {}
