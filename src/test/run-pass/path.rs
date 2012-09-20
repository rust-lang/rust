

mod foo {
    #[legacy_exports];
    fn bar(offset: uint) { }
}

fn main() { foo::bar(0u); }
