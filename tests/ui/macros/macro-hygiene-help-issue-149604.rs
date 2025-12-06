macro_rules! let_it { {} => { let it = (); } }
macro_rules! print_it { {} => { println!("{:?}", it); } }
//~^ ERROR cannot find value `it` in this scope

fn main() {
    let_it!();
    let () = it; //~ ERROR cannot find value `it` in this scope
    print_it!();
}
