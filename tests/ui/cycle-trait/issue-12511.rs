trait T1 : T2 {
//~^ ERROR cycle detected
}
//@ ignore-parallel-frontend query cycle
trait T2 : T1 {
}

fn main() { }
