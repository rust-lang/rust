


// -*- rust -*-
fn ho(fn(int) -> int  f) -> int { let int n = f(3); ret n; }

fn direct(int x) -> int { ret x + 1; }

fn main() {
    let int a =
        direct(3); // direct
                   //let int b = ho(direct); // indirect unbound

    let int c =
        ho(bind direct(_)); // indirect bound
                            //assert (a == b);
                            //assert (b == c);

}