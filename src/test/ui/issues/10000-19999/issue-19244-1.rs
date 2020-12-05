const TUP: (usize,) = (42,);

fn main() {
    let a: [isize; TUP.1];
    //~^ ERROR no field `1` on type `(usize,)`
}
