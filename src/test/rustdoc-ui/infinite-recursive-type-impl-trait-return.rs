// normalize-stderr-test: "`.*`" -> "`DEF_ID`"
// normalize-stdout-test: "`.*`" -> "`DEF_ID`"
// edition:2018

pub async fn f() -> impl std::fmt::Debug {
    #[derive(Debug)]
    enum E {
    //~^ ERROR recursive type `f::{{closure}}#0::E` has infinite size
        This(E),
        Unit,
    }
    E::Unit
}

fn main() {}
