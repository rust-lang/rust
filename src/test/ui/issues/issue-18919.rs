type FuncType<'f> = dyn Fn(&isize) -> isize + 'f;

fn ho_func(f: Option<FuncType>) {
    //~^ ERROR the size for values of type
}

fn main() {}
