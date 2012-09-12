// error-pattern:goodfail

extern mod std;

fn goodfail() {
    task::yield();
    fail ~"goodfail";
}

fn main() {
    task::spawn(|| goodfail() );
    let po = comm::Port();
    // We shouldn't be able to get past this recv since there's no
    // message available
    let i: int = comm::recv(po);
    fail ~"badfail";
}
