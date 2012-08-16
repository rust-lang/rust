// error-pattern:explicit failure

// Just testing unwinding

use std;

fn getbig_and_fail(&&i: int) {
    let _r = and_then_get_big_again(5);
    if i != 0 {
        getbig_and_fail(i - 1);
    } else {
        fail;
    }
}

struct and_then_get_big_again {
  let x:int;
  new(x:int) {self.x = x;}
  drop {
    fn getbig(i: int) {
        if i != 0 {
            getbig(i - 1);
        }
    }
    getbig(100);
  }
}

fn main() {
    do task::spawn {
        getbig_and_fail(400);
    };
}