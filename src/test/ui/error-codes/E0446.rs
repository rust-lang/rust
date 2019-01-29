mod foo {
    struct Bar(u32);

    pub fn bar() -> Bar { //~ ERROR E0446
        Bar(0)
    }
}

fn main() {}
