tag maybe_pointy {
    none;
    p(@pointy);
}

type pointy = {
    mutable a : maybe_pointy,
    c : ~int,
    d : sendfn()->(),
};

fn empty_pointy() -> @pointy {
    ret @{
        mutable a : none,
        c : ~22,
        d : sendfn()->(){},
    }
}

fn main()
{
    let v = empty_pointy();
    v.a = p(v);
}
