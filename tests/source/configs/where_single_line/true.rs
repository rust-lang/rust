// rustfmt-where_single_line: true
// Where style


fn lorem_two_items<Ipsum, Dolor, Sit, Amet>() -> T where Ipsum: Eq, Lorem: Eq {
    // body
}

fn lorem_multi_line<Ipsum, Dolor, Sit, Amet>(
    a: Aaaaaaaaaaaaaaa,
    b: Bbbbbbbbbbbbbbbb,
    c: Ccccccccccccccccc,
    d: Ddddddddddddddddddddddddd,
    e: Eeeeeeeeeeeeeeeeeee,
) -> T
where
    Ipsum: Eq,
{
    // body
}

fn lorem<Ipsum, Dolor, Sit, Amet>() -> T where Ipsum: Eq {
    // body
}

unsafe impl Sync for Foo where (): Send {}
