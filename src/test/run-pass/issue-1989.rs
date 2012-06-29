// exec-env:RUST_CC_ZEAL=1

enum maybe_pointy {
    none,
    p(@pointy)
}

type pointy = {
    mut a : maybe_pointy,
    mut f : fn@()->(),
};

fn empty_pointy() -> @pointy {
    ret @{
        mut a : none,
        mut f : fn@()->(){},
    }
}

fn main()
{
    let v = ~[empty_pointy(), empty_pointy()];
    v[0].a = p(v[0]);
}
