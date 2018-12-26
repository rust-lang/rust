// forbid-output: &mut mut self

struct Struct;

impl Struct {
    fn foo(&mut self) {
        (&mut self).bar(); //~ ERROR cannot borrow
    }

    // In this case we could keep the suggestion, but to distinguish the
    // two cases is pretty hard. It's an obscure case anyway.
    fn bar(self: &mut Self) {
        //~^ WARN function cannot return without recursing
        (&mut self).bar(); //~ ERROR cannot borrow
    }
}

fn main () {}
