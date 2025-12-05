//@ run-rustfix
fn main() {
    {
        break (); //~ ERROR `break` outside of a loop or labeled block
    }
    {
        {
            break (); //~ ERROR `break` outside of a loop or labeled block
        }
    }
}
