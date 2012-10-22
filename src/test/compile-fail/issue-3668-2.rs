fn f(x:int) {
    const child: int = x + 1; //~ ERROR attempt to use a non-constant value in a constant
}

fn main() {}
