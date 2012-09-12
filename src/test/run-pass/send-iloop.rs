// xfail-win32
extern mod std;

fn die() {
    fail;
}

fn iloop() {
    task::spawn(|| die() );
    let p = comm::Port::<()>();
    let c = comm::Chan(p);
    loop {
        // Sending and receiving here because these actions yield,
        // at which point our child can kill us
        comm::send(c, ());
        comm::recv(p);
    }
}

fn main() {
    for uint::range(0u, 16u) |_i| {
        task::spawn_unlinked(|| iloop() );
    }
}
