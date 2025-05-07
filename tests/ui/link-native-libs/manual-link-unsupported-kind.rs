//@ compile-flags:-l raw-dylib=foo

fn main() {
}

//~? ERROR unknown library kind `raw-dylib`
