// xfail-pretty


// -*- rust -*-
fn ho(f: fn(int) -> int ) -> int { let n: int = f(3); ret n; }

fn direct(x: int) -> int { ret x + 1; }

fn main() {
    let a: int =
        direct(3); // direct
                   //let int b = ho(direct); // indirect unbound


    let c: int =
        ho(bind direct(_)); // indirect bound
                            //assert (a == b);
                            //assert (b == c);


}