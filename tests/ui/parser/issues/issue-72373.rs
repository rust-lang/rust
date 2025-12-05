fn foo(c: &[u32], n: u32) -> u32 {
    match *c {
        [h, ..] if h > n => 0,
        [h, ..] if h == n => 1,
        [h, ref ts..] => foo(c, n - h) + foo(ts, n),
        //~^ ERROR expected one of `,`, `@`, `]`, `if`, or `|`, found `..`
        [] => 0,
    }
}
