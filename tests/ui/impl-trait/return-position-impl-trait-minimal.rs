//@ build-pass (FIXME(62277): could be check-pass?)

fn main() {}

fn foo() -> impl std::fmt::Debug { "cake" }
