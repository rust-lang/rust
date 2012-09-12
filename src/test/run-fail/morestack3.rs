// error-pattern:explicit failure

// Just testing unwinding

extern mod std;

fn getbig_and_fail(&&i: int) {
    let _r = and_then_get_big_again(5);
    if i != 0 {
        getbig_and_fail(i - 1);
    } else {
        fail;
    }
}

struct and_then_get_big_again {
  x:int,
  drop {
    fn getbig(i: int) {
        if i != 0 {
            getbig(i - 1);
        }
    }
    getbig(100);
  }
}

fn and_then_get_big_again(x:int) -> and_then_get_big_again {
    and_then_get_big_again {
        x: x
    }
}

fn main() {
    do task::spawn {
        getbig_and_fail(400);
    };
}