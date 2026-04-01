// produce special borrowck message inside all kinds of loops

struct FuncWrapper<'a, T : 'a> {
    func : fn(&'a mut T) -> ()
}

impl<'a, T : 'a> FuncWrapper<'a, T> {
    fn in_loop(self, arg : &'a mut T) {
        loop {
            (self.func)(arg) //~ ERROR cannot borrow
        }
    }

    fn in_while(self, arg : &'a mut T) {
        while true { //~ WARN denote infinite loops with
            (self.func)(arg) //~ ERROR cannot borrow
        }
    }

    fn in_for(self, arg : &'a mut T) {
        let v : Vec<()> = vec![];
        for _ in v.iter() {
            (self.func)(arg) //~ ERROR cannot borrow
        }
    }
}

fn main() {
}
