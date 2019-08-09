struct A {
    banana: u8,
}

impl A {
    fn new(peach: u8) -> A {
        A {
            banana: banana //~ ERROR cannot find value `banana` in this scope
        }
    }

    fn foo(&self, peach: u8) -> A {
        A {
            banana: banana //~ ERROR cannot find value `banana` in this scope
        }
    }
}
