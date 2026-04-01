//@ check-pass
//@ compile-flags: -Wunused

fn foo() -> &'static str { //~ WARN function `foo` is never used
    "hello"
}

fn main() {}
