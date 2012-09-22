

mod foo {
    #[legacy_exports];
    fn bar(offset: uint) { }
}

fn main(args: ~[~str]) { foo::bar(0u); }
