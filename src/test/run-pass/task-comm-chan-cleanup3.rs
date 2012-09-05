use std;

fn main() {
    let c = {
        let p = comm::Port();
        comm::Chan(p)
    };
    comm::send(c, ~"coffee");
}