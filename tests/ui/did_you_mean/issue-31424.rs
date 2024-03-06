//@ forbid-output: &mut mut self

struct Struct;

impl Struct {
    fn foo(&mut self) {
        (&mut self).bar(); //~ ERROR cannot borrow
        //~^ HELP try removing `&mut` here
    }

    // In this case we could keep the suggestion, but to distinguish the
    // two cases is pretty hard. It's an obscure case anyway.
    fn bar(self: &mut Self) {
        //~^ WARN function cannot return without recursing
        //~^^ HELP a `loop` may express intention better if this is on purpose
        (&mut self).bar(); //~ ERROR cannot borrow
        //~^ HELP try removing `&mut` here
    }
}

fn main () {}
