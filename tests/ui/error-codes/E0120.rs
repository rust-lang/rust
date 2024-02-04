trait MyTrait { fn foo() {} }

impl Drop for dyn MyTrait {
              //~^ ERROR E0120
    fn drop(&mut self) {}
              //~^ ERROR E0038

}

fn main() {}
