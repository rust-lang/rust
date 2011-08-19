// Export the tag variants, without the tag

mod foo {
    export t1;
    tag t { t1; }
}

fn main() { let v = foo::t1; }
