use std::collections::HashSet;

struct Foo {
  n: HashSet<isize>,
}

impl Foo {
    pub fn foo<F>(&mut self, mut fun: F) where F: FnMut(&isize) {
        for f in &self.n {
            fun(f);
        }
    }
}

fn bar(f: &mut Foo) {
    f.foo(
    //~^ ERROR cannot borrow `*f` as mutable
        |a| { //~ ERROR closure requires unique access to `f`
            f.n.insert(*a);
        })
}

fn main() {
  let mut f = Foo { n: HashSet::new() };
  bar(&mut f);
}
