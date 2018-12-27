#![feature(decl_macro)]

mod foo {
    pub fn f() {}
}

mod bar {
    pub fn g() {}
}

macro m($($t:tt)*) {
    $($t)*
    use foo::*;
    f();
    g(); //~ ERROR cannot find function `g` in this scope
}

fn main() {
    m! {
        use bar::*;
        g();
        f(); //~ ERROR cannot find function `f` in this scope
    }
}

n!(f);
macro n($i:ident) {
    mod foo {
        pub fn $i() -> u32 { 0 }
        pub fn f() {}

        mod test {
            use super::*;
            fn g() {
                let _: u32 = $i();
                let _: () = f();
            }
        }

        macro n($j:ident) {
            mod test {
                use super::*;
                fn g() {
                    let _: u32 = $i();
                    let _: () = f();
                    $j();
                }
            }
        }
        macro n_with_super($j:ident) {
            mod test {
                use super::*;
                fn g() {
                    let _: u32 = $i();
                    let _: () = f();
                    super::$j();
                }
            }
        }

        n!(f); //~ ERROR cannot find function `f` in this scope
        n_with_super!(f);
        mod test2 {
            super::n! {
                f //~ ERROR cannot find function `f` in this scope
            }
            super::n_with_super! {
                f
            }
        }
    }
}
