extern "C" {
    fn foo() -> i32 { //~ ERROR incorrect function inside `extern` block
        return 0;
    }
}

extern "C" fn bar() -> i32 {
    return 0;
}

fn main() {}
