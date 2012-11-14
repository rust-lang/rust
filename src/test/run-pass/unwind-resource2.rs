// xfail-win32
extern mod std;

struct complainer {
  c: @int,
}

impl complainer : Drop {
    fn finalize() {}
}

fn complainer(c: @int) -> complainer {
    complainer {
        c: c
    }
}

fn f() {
    let c = move complainer(@0);
    fail;
}

fn main() {
    task::spawn_unlinked(f);
}
