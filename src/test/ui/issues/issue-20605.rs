fn changer<'a>(mut things: Box<Iterator<Item=&'a mut u8>>) {
    for item in *things { *item = 0 }
//~^ ERROR the size for values of type
}

fn main() {}
