// error-pattern:explicit failure
// Issue #2272 - unwind this without leaking the unique pointer

fn main() {
    let _x = {
        y: {
            z: @0
        },
        a: ~0
    };
    fail;
}