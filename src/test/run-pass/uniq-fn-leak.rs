// xfail-test
    tag maybe_pointy {
        none;
        p(@pointy);
    }

    type pointy = {
        mutable a : maybe_pointy,
        d : sendfn()->(),
    };

    fn empty_pointy() -> @pointy {
        ret @{
            mutable a : none,
            d : sendfn()->(){},
        }
    }

    fn main()
    {
        let v = empty_pointy();
        v.a = p(v);
    }
