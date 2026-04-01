pub use foo::FOO2;

pub const FOO: usize = 3;
const BAR: usize = 3;

mod foo {
    pub const FOO2: usize = 3;
}
