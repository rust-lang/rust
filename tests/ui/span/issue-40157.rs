fn main () {
    {println!("{:?}", match { let foo = vec![1, 2]; foo.get(1) } { x => x });}
     //~^ ERROR does not live long enough
}
