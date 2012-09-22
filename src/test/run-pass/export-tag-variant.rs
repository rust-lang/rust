// Export the enum variants, without the enum

mod foo {
    #[legacy_exports];
    export t1;
    enum t { t1, }
}

fn main() { let v = foo::t1; }
