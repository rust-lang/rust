struct S {
    x: [usize; 3],
}

fn foo() {
    {
        {
            println!("hi");
        }
    }
}

fn main() {
//~^ NOTE unclosed delimiter
    {
        {
        //~^ NOTE this delimiter might not be properly closed...
            foo();
    }
    //~^ NOTE ...as it matches this but it has different indentation
}
//~ ERROR this file contains an unclosed delimiter
