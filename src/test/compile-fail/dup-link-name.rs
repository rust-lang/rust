// xfail-stage0
// error-pattern:duplicate meta item `name`

#[link(name = "test",
       name)];

fn main() {}