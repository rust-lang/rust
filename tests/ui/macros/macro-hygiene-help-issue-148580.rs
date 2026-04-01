macro_rules! print_it { {} => { println!("{:?}", it); } }
//~^ ERROR cannot find value `it` in this scope

fn main() {
    {
        let it = "hello";
    }
    {
        let it = "world";
        {
            let it = ();
            print_it!();
        }
    }
}
