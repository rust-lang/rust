

fn foo() {
    fn zed(z: bar) { }
    tag bar { nil; }
    fn baz() { zed(nil); }
}

fn main(args: vec[str]) { }