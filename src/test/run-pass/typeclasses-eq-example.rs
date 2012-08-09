// Example from lkuper's intern talk, August 2012.

trait Equal {
    fn isEq(a: self) -> bool;
}

enum Color { cyan, magenta, yellow, black }

impl Color : Equal {
    fn isEq(a: Color) -> bool {
        match (self, a) {
          (cyan, cyan)       => { true  }
          (magenta, magenta) => { true  }
          (yellow, yellow)   => { true  }
          (black, black)     => { true  }
          _                  => { false }
        }
    }
}

enum ColorTree {
    leaf(Color),
    branch(@ColorTree, @ColorTree)
}

impl ColorTree : Equal {
    fn isEq(a: ColorTree) -> bool {
        match (self, a) {
          (leaf(x), leaf(y)) => { x.isEq(y) }
          (branch(l1, r1), branch(l2, r2)) => { 
            (*l1).isEq(*l2) && (*r1).isEq(*r2)
          }
          _ => { false }
        }
    }
}

fn main() {
    assert cyan.isEq(cyan);
    assert magenta.isEq(magenta);
    assert !cyan.isEq(yellow);
    assert !magenta.isEq(cyan);

    assert leaf(cyan).isEq(leaf(cyan));
    assert !leaf(cyan).isEq(leaf(yellow));

    assert branch(@leaf(magenta), @leaf(cyan))
        .isEq(branch(@leaf(magenta), @leaf(cyan)));

    assert !branch(@leaf(magenta), @leaf(cyan))
        .isEq(branch(@leaf(magenta), @leaf(magenta)));

    log(error, "Assertions all succeeded!");
}