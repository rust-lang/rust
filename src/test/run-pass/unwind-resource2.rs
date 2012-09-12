// xfail-win32
extern mod std;

struct complainer {
  c: @int,
  drop {}
}

fn complainer(c: @int) -> complainer {
    complainer {
        c: c
    }
}

fn f() {
    let c <- complainer(@0);
    fail;
}

fn main() {
    task::spawn_unlinked(f);
}
