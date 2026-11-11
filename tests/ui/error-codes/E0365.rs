mod foo {
    pub const X: u32 = 1;
}

pub use foo as foo2;
//~^ ERROR `foo` is only public within the crate, and cannot be re-exported outside [E0365]

fn main() {}
