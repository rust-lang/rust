fn dup(x: Box<isize>) -> Box<(Box<isize>,Box<isize>)> {


    Box::new((x, x))
    //~^ use of moved value: `x` [E0382]
}

fn main() {
    dup(Box::new(3));
}
