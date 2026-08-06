//@ normalize-stderr: "`.*`" -> "`DEF_ID`"
//@ normalize-stdout: "`.*`" -> "`DEF_ID`"
//@ edition:2018

pub async fn f() -> impl std::fmt::Debug {
    #[derive(Debug)]
    enum E { //~ ERROR
        This(E),
        Unit,
    }
    E::Unit
}

fn main() {}
