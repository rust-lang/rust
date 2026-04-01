trait MyTrait { fn foo(&self) {} }

impl Drop for dyn MyTrait {
              //~^ ERROR E0120
    fn drop(&mut self) {}

}

fn main() {}
