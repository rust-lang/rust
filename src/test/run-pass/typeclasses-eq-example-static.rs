// Example from lkuper's intern talk, August 2012 -- now with static
// methods!

trait Equal {
    static fn isEq(a: self, b: self) -> bool;
}

enum Color { cyan, magenta, yellow, black }

impl Color : Equal {
    static fn isEq(a: Color, b: Color) -> bool {
        match (a, b) {
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
    static fn isEq(a: ColorTree, b: ColorTree) -> bool {
        match (a, b) {
          (leaf(x), leaf(y)) => { isEq(x, y) }
          (branch(l1, r1), branch(l2, r2)) => { 
            isEq(*l1, *l2) && isEq(*r1, *r2)
          }
          _ => { false }
        }
    }
}

fn main() {
    assert isEq(cyan, cyan);
    assert isEq(magenta, magenta);
    assert !isEq(cyan, yellow);
    assert !isEq(magenta, cyan);

    assert isEq(leaf(cyan), leaf(cyan));
    assert !isEq(leaf(cyan), leaf(yellow));

    assert isEq(branch(@leaf(magenta), @leaf(cyan)),
                branch(@leaf(magenta), @leaf(cyan)));

    assert !isEq(branch(@leaf(magenta), @leaf(cyan)),
                 branch(@leaf(magenta), @leaf(magenta)));

    log(error, "Assertions all succeeded!");
}