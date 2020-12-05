fn changer<'a>(mut things: Box<dyn Iterator<Item=&'a mut u8>>) {
    for item in *things { *item = 0 }
//~^ ERROR the size for values of type
}

fn main() {}
