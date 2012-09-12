// xfail-win32 leaks
extern mod std;

fn die() {
    fail;
}

fn iloop() {
    task::spawn(|| die() );
}

fn main() {
    for uint::range(0u, 100u) |_i| {
        task::spawn_unlinked(|| iloop() );
    }
}
