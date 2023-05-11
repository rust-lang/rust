// Regression test for #63033.

// check-pass
// edition: 2018

async fn test1(_: &'static u8, _: &'_ u8, _: &'_ u8) {}

async fn test2<'s>(_: &'s u8, _: &'_ &'s u8, _: &'_ &'s u8) {}

fn main() {}
