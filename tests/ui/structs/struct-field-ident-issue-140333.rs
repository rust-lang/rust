fn a() -> impl b< //~ ERROR cannot find trait `b` in this scope
    [c; { //~ ERROR cannot find type `c` in this scope
    struct D {
            #[a] //~ ERROR annot find attribute `a` in this scope
            bar: e, //~ ERROR cannot find type `e` in this scope
        }
    }],
> {
    todo!("need to implement")
}

fn main() {}
