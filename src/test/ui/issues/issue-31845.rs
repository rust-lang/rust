// Checks lexical scopes cannot see through normal module boundaries

fn f() {
    fn g() {}
    mod foo {
        fn h() {
           g(); //~ ERROR cannot find function `g` in this scope
        }
    }
}

fn main() {}
