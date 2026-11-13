fn foo<const X: u32>() {
    fn bar() -> u32 {
        X //~ ERROR can't use generic parameters from outer item
    }
}

fn main() {}
