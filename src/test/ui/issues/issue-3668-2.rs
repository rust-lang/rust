fn f(x:isize) {
    static child: isize = x + 1;
    //~^ ERROR can't capture dynamic environment
}

fn main() {}
